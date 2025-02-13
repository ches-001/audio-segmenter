import os
import yaml
import time
import tqdm
import math
import logging
import torch
import pandas as pd
from datetime import datetime
from modules.net import AudioSegmentationNet
from matplotlib import pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from utils import ddp_sync_metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import *

LOGGER = logging.getLogger(__name__)

class TrainAudioSegPipeline:
    metrics_dir = "metrics/audio_segmentation"
    checkpoints_dir = "saved_model/audio_segmentation/checkpoints"
    best_model_dir = "saved_model/audio_segmentation/best_model"

    def __init__(
        self, 
        model: AudioSegmentationNet,
        loss_fn: Any,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]=None,
        lr_schedule_interval: int=1,
        model_name: Optional[str]=None,
        checkpoint_path: Optional[str]=None,
        device_or_rank: Union[int, str]="cpu",
        ddp_mode: bool=False,
        config_path: str="config/config.yaml"
    ):
        LOGGER.info(f"Number of model paramters: {sum([i.numel() for i in model.parameters()])}")
        self.model                = model
        self.model.to(device_or_rank)
        self.loss_fn              = loss_fn
        self.optimizer            = optimizer
        self.lr_scheduler         = lr_scheduler
        self.lr_schedule_interval = lr_schedule_interval
        self.model_name           = model_name or self.model.__class__.__name__
        self.device_or_rank       =  device_or_rank
        self.ddp_mode             = ddp_mode
        self.last_epoch           = 0

        if self.ddp_mode:
            self.model = DDP(self.model, device_ids=[self.device_or_rank, ])
            LOGGER.info(f"Model copied to device: {self.device_or_rank} for distributed training")
        self.checkpoints_dir = os.path.join(self.checkpoints_dir, str(int(time.time())))

        self._save_config_copy(config_path, to_checkpoint_dir=True) # save config to checkpoint_dir
        self._save_config_copy(config_path, to_checkpoint_dir=False) # save config to best_model_dir

        # collect metrics in this list of dicts
        self._train_metrics: List[Dict[str, float]] = []
        self._eval_metrics: List[Dict[str, float]] = []

        # load checkpoint if any
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
    
    def _save(self, path: str, snapshot_mode: bool=True):
        num_classes = self.model.module.num_classes if isinstance(self.model, DDP) else self.model.num_classes
        network_params = self.model.module.state_dict() if isinstance(self.model, DDP) else self.model.state_dict()
        if snapshot_mode:
            state_dicts = {
                "LAST_EPOCH": self.last_epoch,
                "NETWORK_PARAMS": network_params,
                "OPTIMIZER_PARAMS": self.optimizer.state_dict(),
                "METRICS": {
                    "TRAIN": self._train_metrics,
                    "EVAL": self._eval_metrics
                },
                "NUM_CLASSES": num_classes
            }
            if self.lr_scheduler:
                state_dicts["LR_SCHEDULER_PARAMS"] = self.lr_scheduler.state_dict()
        else:
            state_dicts = {
                "LAST_EPOCH": self.last_epoch,
                "NETWORK_PARAMS": network_params,
                "NUM_CLASSES": num_classes
            }
        return torch.save(state_dicts, path)
    
    def _save_config_copy(self, config_path: str, to_checkpoint_dir: bool):
        if not (self.ddp_mode) or (self.ddp_mode and self.device_or_rank in [0, "cuda:0"]):
            dest_path = os.path.join(
                (self.checkpoints_dir if to_checkpoint_dir else self.best_model_dir), "config"
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            f.close()
            os.makedirs(dest_path, exist_ok=True)
            with open(os.path.join(dest_path, "config.yaml"), "w") as f:
                yaml.safe_dump(config, f, sort_keys=False, default_flow_style=True)
            f.close()

    def save_best_model(self):
        # If DDP mode, we only need to ensure that the model is only being saved at
        # rank-0 device. It does not neccessarily matter the rank, as long as the
        # model saving only happens on one rank only (one device) since the model
        # is exactly the same across all
        if (not self.ddp_mode) or (self.ddp_mode and self.device_or_rank in [0, "cuda:0"]):
            if not os.path.isdir(self.best_model_dir): 
                os.makedirs(self.best_model_dir, exist_ok=True)
            path = os.path.join(self.best_model_dir, f"{self.model_name}.pth.tar")
            self._save(path, snapshot_mode=False)

    def save_checkpoint(self):
        # similar to the `save_best_model` method, the model for only one device needs
        # to be saved.
        if (not self.ddp_mode) or (self.ddp_mode and self.device_or_rank in [0, "cuda:0"]):
            if not os.path.isdir(self.checkpoints_dir): 
                os.makedirs(self.checkpoints_dir, exist_ok=True)
            time = str(datetime.now())
            time = time.replace(":", "-")
            path = os.path.join(self.checkpoints_dir, f"{self.model_name}-{self.last_epoch}-{time}.pth.tar")
            self._save(path, snapshot_mode=True)
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        # if DDP mode, we do not need to only do this for rank-0 devices but
        # across all devices to ensure that the model and optimizer states 
        # begin at same point
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist")
        saved_states = torch.load(path, map_location=self.device_or_rank)
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(saved_states["NETWORK_PARAMS"])
        else:
            self.model.load_state_dict(saved_states["NETWORK_PARAMS"])
        self.optimizer.load_state_dict(saved_states["OPTIMIZER_PARAMS"])
        if self.lr_scheduler and "LR_SCHEDULER_PARAMS" in saved_states.keys():
            self.lr_scheduler.load_state_dict(saved_states["LR_SCHEDULER_PARAMS"])
        self.last_epoch     = saved_states["LAST_EPOCH"]
        self._train_metrics = saved_states["METRICS"]["TRAIN"]
        self._eval_metrics  = saved_states["METRICS"]["EVAL"]
        return saved_states
    
    def train(self, dataloader: DataLoader, verbose: bool=False, max_steps: Optional[int]=None) -> Dict[str, float]:
        r = self.step(dataloader, "train", verbose, max_steps=max_steps)
        if self.lr_scheduler and (self.last_epoch % self.lr_schedule_interval == 0):
            self.lr_scheduler.step()
        self.last_epoch += 1
        return r
        
    def evaluate(self, dataloader: DataLoader, verbose: bool=False, max_steps: Optional[int]=None) -> Dict[str, float]:        
        with torch.no_grad():
            return self.step(dataloader, "eval", verbose, max_steps=max_steps)

    def step(
            self, 
            dataloader: DataLoader, 
            mode: str, 
            verbose: bool=False, 
            max_steps: Optional[int]=None
        ) -> Dict[str, float]:
        if mode not in self._valid_modes:
            raise ValueError(f"Invalid mode {mode} expected either one of {self._valid_modes}")
        getattr(self.model, mode)()

        metrics = {}

        if self.ddp_mode:
            # invert progress bar position such that the last (rank n-1) is at
            # the top and the first (rank 0) at the bottom. This is because the
            # first rank will be the one logging all the metrics
            world_size = int(os.environ["WORLD_SIZE"])
            position   = (
                self.device_or_rank 
                if not isinstance(self.device_or_rank, str) else 
                int(self.device_or_rank.replace("cuda:", ""))
            )
            position   = abs(position - (world_size - 1))
            pbar = tqdm.tqdm(enumerate(dataloader), position=position)
        else:
            pbar = tqdm.tqdm(enumerate(dataloader))

        for count, (signals, classes, _) in pbar:
            if max_steps is not None and count + 1 > max_steps:
                break
            signals: torch.Tensor     = signals.to(self.device_or_rank)
            classes: torch.Tensor     = classes.to(self.device_or_rank, dtype=torch.int64).squeeze()
            pred_logits: torch.Tensor = self.model(signals)
            batch_loss: torch.Tensor  = self.loss_fn(pred_logits, classes)

            if mode == "train":
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            batch_metrics              = {}
            targets                    = classes.cpu().numpy()
            pred                       = pred_logits.argmax(dim=-1).detach().cpu().numpy()
            batch_metrics["loss"]      = batch_loss.item()
            batch_metrics["accuracy"]  = accuracy_score(targets, pred)
            batch_metrics["precision"] = precision_score(targets, pred)
            batch_metrics["recall"]    = recall_score(targets, pred)
            batch_metrics["f1"]        = f1_score(targets, pred)

            # sum metrics across all batches
            for key in batch_metrics.keys(): 
                if key not in metrics.keys(): 
                    metrics[key] = batch_metrics[key]
                else: metrics[key] += batch_metrics[key]
        
        # average metrics from all batches
        for key in metrics.keys(): 
            metrics[key] = (metrics[key] / (count + 1))
        
        # sync metrics across all devices if ddp_mode=True
        if self.ddp_mode:
            metrics = ddp_sync_metrics(self.device_or_rank, metrics)

        if not self.ddp_mode or (self.ddp_mode and self.device_or_rank in [0, f"cuda:0"]):
            getattr(self, f"_{mode}_metrics").append(metrics)
            if verbose:
                log = "[" + mode.title() + "]: " + "\t".join([f"{k.replace('_', ' ')}: {v :.4f}" for k, v in metrics.items()])
                print(log)
        return metrics
    
    def save_metrics_plots(self, figsize: Tuple[float, float]=(15, 60)):
        # metrics ought to be saved by only one device process since they will be collected
        # and synced across all devices involved
        if not self.ddp_mode or (self.ddp_mode and self.device_or_rank in [0, f"cuda:0"]):
            self._save_metrics_plots("train", figsize)
            self._save_metrics_plots("eval", figsize)
    
    def metrics_to_csv(self):
        if not self.ddp_mode or (self.ddp_mode and self.device_or_rank in [0, f"cuda:0"]):
            if not os.path.isdir(self.metrics_dir): 
                os.makedirs(self.metrics_dir, exist_ok=True)
            pd.DataFrame(self._train_metrics).to_csv(os.path.join(self.metrics_dir, "train_metrics.csv"), index=False)
            pd.DataFrame(self._eval_metrics).to_csv(os.path.join(self.metrics_dir, "eval_metrics.csv"), index=False)

    def _save_metrics_plots(self, mode: str, figsize: Tuple[float, float]=(15, 60)):        
        valid_modes = self._valid_modes
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")
        
        df       = pd.DataFrame(getattr(self, f"_{mode}_metrics"))
        nrows    = len(df.columns)
        fig, axs = plt.subplots(nrows, 1, figsize=figsize)

        if nrows == 1:
            label = df.columns[0]
            axs.plot(df[df.columns[0]].to_numpy())
            axs.grid(visible=True)
            axs.set_xlabel("Epoch")
            axs.set_ylabel(label)
            axs.set_title(f"[{mode.title()}] {label} vs Epoch", fontsize=24)
            axs.tick_params(axis='x', which='major', labelsize=20)
        else:
            for i, col in enumerate(df.columns):
                label = col.replace("_", " ").title()
                axs[i].plot(df[col].to_numpy())
                axs[i].grid(visible=True)
                axs[i].set_xlabel("Epoch")
                axs[i].set_ylabel(label)
                axs[i].set_title(f"[{mode.title()}] {label} vs Epoch", fontsize=24)
                axs[i].tick_params(axis='x', which='major', labelsize=20)

        if os.path.isdir(self.metrics_dir): os.makedirs(self.metrics_dir, exist_ok=True)
        fig.savefig(os.path.join(self.metrics_dir, f"{mode}_metrics_plot.jpg"))
        fig.clear()
        plt.close(fig)

    @property
    def _valid_modes(self) -> Iterable[str]:
        return ["train", "eval"]