import warnings
warnings.filterwarnings(action="ignore")
import os
import sys
import tqdm
import argparse
import logging
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Sampler, DistributedSampler
from dataset import AudioIterableTrainDataset, AudioDataset
from modules.net import AudioSegmentationNet
from trainer.trainer import TrainAudioSegPipeline
from utils import load_yaml, load_json, save_json, ddp_setup, ddp_destroy, ddp_broadcast
from typing import *

LOGGER      = logging.getLogger(__name__)
CONFIG_PATH = "config/config.yaml"


def generate_annotations(data_dir: str, annotator: str) -> List[pd.DataFrame]:
    train_path  = os.path.join(data_dir, "train")
    eval_path   = os.path.join(data_dir, "eval")
    config      = load_yaml(CONFIG_PATH)
    annotations = load_json(os.path.join(data_dir, "annotations", "annotation.json"))
    annotations = annotations["annotations"][annotator]
    dfs         = []
    for path, name in zip([train_path, eval_path], ["train", "eval"]):
        dataset     = AudioIterableTrainDataset(path, annotations, config, ignore_sample_error=True, only_labels=True)
        df = pd.DataFrame([sample for sample in tqdm.tqdm(dataset)])
        df.to_csv(os.path.join(data_dir, "annotations", f"{name}_annotation.csv"), index=False)
        dfs.append(df)
    return dfs


def make_dataset(
        data_dir: str, 
        csv_annotations: Dict[str, Any],
        config: Dict[str, Any],
        ignore_sample_error: bool,
        ext: str="wav",
    ) -> AudioDataset:

    dataset = AudioDataset(
        data_dir, csv_annotations, config, ext=ext, ignore_sample_error=ignore_sample_error
    )
    return dataset


def make_dataloader(
        dataset: AudioDataset, 
        batch_size: int, 
        sampler: Optional[Sampler]=None, 
        **kwargs
    ) -> DataLoader:
    
    kwargs = dict(batch_size=batch_size, **kwargs)

    if "num_workers" not in kwargs:
        kwargs["num_workers"] = os.cpu_count()

    dataloader = DataLoader(
        dataset, sampler=sampler, shuffle=(True if sampler is None else None), **kwargs
    )
    return dataloader


def make_model(
        num_classes: int, 
        sample_rate: int,
        config: Dict[str, Any]
    ) -> AudioSegmentationNet:

    model = AudioSegmentationNet(
        num_classes=num_classes,
        sample_rate=sample_rate,
        config=config,
    )
    model.train()
    return model


def make_optimizer(
        model: AudioSegmentationNet, 
        config: Dict[str, Any], 
        ddp_mode: bool=False
    ) -> torch.optim.Optimizer:

    optim_config   = config["train_config"]["optimizer_config"].copy()
    optimizer_name = optim_config.pop("name")

    if ddp_mode:
        optim_config["lr"] *= torch.cuda.device_count()

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optim_config)
    return optimizer


def make_lr_scheduler(
        optimizer: torch.optim.Optimizer, 
        config: Dict[str, Any]
    ) -> torch.optim.lr_scheduler.LRScheduler:

    scheduler_config = config["train_config"]["lr_scheduler_config"].copy()
    scheduler_name   = scheduler_config.pop("name")
    lr_scheduler     = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **scheduler_config)
    return lr_scheduler


def run(args: argparse.Namespace, config: Dict[str, Any]):
    if args.use_ddp:
        # setup DDP process group
        ddp_setup()
        
    data_dir              = args.data_dir
    train_path            = os.path.join(data_dir, "train")
    eval_path             = os.path.join(data_dir, "eval")
    device_or_rank        = config["train_config"]["device"] if torch.cuda.is_available() else "cpu"
    train_annotation_path = os.path.join(data_dir, "annotations", "train_annotation.csv")
    eval_annotation_path  = os.path.join(data_dir, "annotations", "eval_annotation.csv")
    
    if os.path.isfile(train_annotation_path) and os.path.isfile(eval_annotation_path): 
        train_annotations = pd.read_csv(train_annotation_path)
        eval_annotations  = pd.read_csv(eval_annotation_path)
    else:
        train_annotations, eval_annotations = generate_annotations(data_dir, args.annotator)

    train_dataset = make_dataset(train_path, train_annotations, config, args.ignore_sample_error, ext=args.ext)
    eval_dataset  = make_dataset(eval_path, eval_annotations, config, args.ignore_sample_error, args.ext)
    train_sampler = None
    eval_sampler  = None

    if args.use_ddp:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler  = DistributedSampler(eval_dataset)
        try:
            device_or_rank = int(os.environ["LOCAL_RANK"])
        except KeyError as e:
            LOGGER.error(
                f"{e}. This LOCAL_RANK key not existing in the environment variable is a clear "
                "indication that you need to execute this script with torchrun if you wish to "
                "use the DDP mode (ddp=True)"
            )
            sys.exit(0)

    def print_logs(log: str, rank_to_log: Union[str, int]=-1):
        if not args.no_verbose:
            if args.use_ddp:
                if rank_to_log != -1:
                    if device_or_rank == rank_to_log:
                        print(log)
                else:
                    print(log)
                return
            print(log)

    num_workers      = config["train_config"]["num_workers"]
    train_dataloader = make_dataloader(train_dataset, args.batch_size, train_sampler, num_workers=num_workers)
    eval_dataloader  = make_dataloader(eval_dataset, args.batch_size, eval_sampler, num_workers=num_workers)
    
    if not args.use_ddp or (args.use_ddp and device_or_rank in [0, "cuda:0"]):
        class_weights = train_dataset.get_class_weights(device_or_rank)
        num_classes   = torch.tensor(class_weights.shape[0], dtype=torch.int64, device=device_or_rank)
    else:
        num_classes = torch.tensor([0], dtype=torch.int64, device=device_or_rank)

    if args.use_ddp:
        ddp_broadcast(num_classes, src_rank=0)
        if device_or_rank not in [0, "cuda:0"]:
            class_weights = torch.zeros(num_classes.item(), dtype=torch.int64, device=device_or_rank)
        ddp_broadcast(class_weights, src_rank=0)

    sample_rate  = train_dataset.sample_rate
    idx2class    = train_dataset.idx2class
    num_classes  = class_weights.shape[0]
    model        = make_model(num_classes, sample_rate, config)
    loss_fn      = nn.CrossEntropyLoss(weight=class_weights.to(device_or_rank))
    optimizer    = make_optimizer(model, config)
    lr_scheduler = make_lr_scheduler(optimizer, config) if args.lr_schedule else None
    pipeline     = TrainAudioSegPipeline(
        model, 
        loss_fn, 
        optimizer, 
        lr_scheduler=lr_scheduler,
        lr_schedule_interval=args.lr_schedule_interval,
        device_or_rank=device_or_rank,
        ddp_mode=args.use_ddp,
        config_path=CONFIG_PATH
    )

    if not args.use_ddp or (args.use_ddp and device_or_rank in [0, "cuda:0"]):
        save_json(idx2class, os.path.join(pipeline.best_model_dir, "classes.json"))
        save_json(idx2class, os.path.join(pipeline.checkpoints_dir, "classes.json"))

    best_model_epoch = None
    best_loss        = np.inf
    last_epoch       = pipeline.last_epoch

    for epoch in range(last_epoch, args.epochs):
        print_logs(f"train step @ epoch: {epoch} on device: {device_or_rank}", -1)
        _ = pipeline.train(train_dataloader, verbose=(not args.no_verbose))

        if epoch % args.eval_interval == 0:
            print_logs(f"evaluation step @ epoch: {epoch} on device: {device_or_rank}", -1)
            eval_metrics = pipeline.evaluate(eval_dataloader, verbose=(not args.no_verbose))

            if eval_metrics["loss"] < best_loss:
                best_model_epoch = epoch
                best_loss = eval_metrics["loss"]
                pipeline.save_best_model()
                print_logs(f"best model saved at epoch {best_model_epoch}", 0)

        if (args.checkpoint_interval > 0) and (epoch % args.checkpoint_interval == 0):
            print_logs(f"checkpoint saved at epoch: {best_model_epoch}", 0)
            pipeline.save_checkpoint()

    pipeline.metrics_to_csv()
    pipeline.save_metrics_plots()
    print_logs(f"\nBest model saved at epoch {best_model_epoch} with loss value of {best_loss :.4f}", 0)

    if args.use_ddp:
        # Destroy process group
        ddp_destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Detection Network")
    parser.add_argument("--data_dir", type=str, metavar="", required=True, help="Dataset directory")
    parser.add_argument("--annotator", type=str, default="annotator_a", metavar="", help="Annotator label to use")
    parser.add_argument("--ext", type=str, default="wav", metavar="", help="Training data extension (eg: wav, mp3...)")
    parser.add_argument("--batch_size", type=int, default=128, metavar="", help="Training batch size")
    parser.add_argument("--epochs", type=int, default=200, metavar="", help="Number of training epochs")
    parser.add_argument("--checkpoint_interval", type=int, default=10, metavar="", help="Number of epochs before persisting checkpoint to disk")
    parser.add_argument("--eval_interval", type=int, default=1, metavar="", help="Number of training steps before each evaluation")
    parser.add_argument("--no_verbose", action="store_true", help="Reduce training output verbosity")
    parser.add_argument("--lr_schedule", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--ignore_sample_error", action="store_true", help="This ignores errors caused by faulty annotations and logs a warning instead. PS: If you get a lot of warnings in a single epoch, you really need to check your annotations")
    parser.add_argument("--use_ddp", action="store_true", help="Use DDP (Distributed Data Parallelization)")
    parser.add_argument("--lr_schedule_interval", type=int, default=1, metavar="", help="Number of training steps before lr scheduling")

    args            = parser.parse_args()
    SEED            = 42
    LOG_FORMAT      = "%(asctime)s %(levelname)s %(filename)s: %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    config          = load_yaml(CONFIG_PATH)

    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark     = False
        torch.backends.cudnn.deterministic = True

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    
    # For single GPU / CPU training:: train.py --data_dir=<data/directory> --use_ddp --lr_schedule --batch_size=256 --ignore_sample_error
    # For multiple GPU training:: torchrun --standalone --nproc_per_node=gpu train.py --use_ddp --lr_schedule --batch_size=256 --ignore_sample_error
    run(args, config)