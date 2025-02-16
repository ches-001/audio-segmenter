import tqdm
import warnings
warnings.filterwarnings(action="ignore")
import os
import pathlib
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from dataset import AudioInferenceDataset
from modules.net import AudioSegmentationNet
from utils import load_yaml, load_json, save_json
from typing import *

LOGGER = logging.getLogger(__name__)

def smoothen_labels(labels: torch.Tensor, mode_context: int=20, min_cs_window: int=300):
    # this function does an in-place operation on labels tensor
    for i in range(0, labels.shape[0]):
        start     = max(0, i-mode_context)
        end       = min(labels.shape[0], i + mode_context + 1)
        labels[i] = labels[start : end].mode().values

    for i in range(min_cs_window, labels.shape[0]):
        if labels[i] != labels[i-1]:
            if (labels[i-min_cs_window : i] == labels[i]).sum() < (min_cs_window // 2):
                labels[i] = labels[i-1]


def running_length_encoding(
        timeframes: torch.Tensor, 
        labels: torch.Tensor, 
        idx2class: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
    final_pred = []
    for i in range(0, labels.shape[0]):
        class_ = idx2class[str(int(labels[i].item()))]
        if i == 0:
            final_pred.append({
                "start": timeframes[i][0].item(), 
                "end": timeframes[i][1].item(), 
                "class": class_
            })
            continue

        if class_ == final_pred[-1]["class"]:
            final_pred[-1]["end"] = timeframes[i][1].item()
        else:
            final_pred.append({
                "start": timeframes[i][0].item(), 
                "end": timeframes[i][1].item(), 
                "class": class_
            })
    return final_pred
        

def run(args: argparse.Namespace):
    output_dir  = "outputs"
    filename    = pathlib.Path(args.file_path).parts[-1]
    base_path_  = os.path.join(*pathlib.Path(args.model_path).parts[:3])
    config      = load_yaml(os.path.join(base_path_, "config", "config.yaml"))
    idx2class   = load_json(os.path.join(base_path_, "classes.json"))
    device      = config["train_config"]["device"] if torch.cuda.is_available() else "cpu"
    num_classes = len(idx2class)
    dataset     = AudioInferenceDataset(args.file_path, config)
    dataloader  = DataLoader(dataset, batch_size=args.batch_size)
    model       = AudioSegmentationNet(num_classes, dataset.audio_metadata.sample_rate, config)
    state_dict  = torch.load(args.model_path, map_location=device)["NETWORK_PARAMS"]

    model.to(device)
    if hasattr(model, "resampler") and "resampler.kernel" not in state_dict.keys():
        resampler_state_dict =  model.resampler.state_dict()
        if "kernel" in resampler_state_dict.keys():
            state_dict["resampler.kernel"] = model.resampler.state_dict()["kernel"]
        
    model.init_zeros_taper_window(state_dict["taper_window"])
    model.load_state_dict(state_dict)
    model.eval()
    
    labels     = []
    timeframes = []
    for batch_signals, batch_timeframes in tqdm.tqdm(dataloader):
        batch_signals = batch_signals.to(device)
        with torch.inference_mode():
            logits: torch.Tensor       = model(batch_signals)
            batch_labels: torch.Tensor = logits.argmax(dim=1)
            labels.append(batch_labels)
            timeframes.append(batch_timeframes)

    labels     = torch.concat(labels, dim=0)
    timeframes = torch.concat(timeframes, dim=0)

    smoothen_labels(labels, args.mode_context, args.min_cs_window)
    final_labels = running_length_encoding(timeframes, labels, idx2class)
    os.makedirs(output_dir, exist_ok=True)
    save_json(final_labels, os.path.join(output_dir, ".".join(filename.split(".")[:-1])) + ".json")


if __name__ == "__main__":
    model_path = f"saved_model/audio_segmentation/best_model/{AudioSegmentationNet.__name__}.pth.tar"

    parser = argparse.ArgumentParser(description="Train Detection Network")
    parser.add_argument("--file_path", type=str, metavar="", required=True, help="Audio file path to run inference on")
    parser.add_argument("--model_path", type=str, default=model_path,metavar="", help="Model path")
    parser.add_argument("--batch_size", type=int, default=128, metavar="", help="Training batch size")
    parser.add_argument("--mode_context", type=int, default=20, metavar="", help="Mode context for label smoothening")
    parser.add_argument("--min_cs_window", type=int, default=300, metavar="", help="Minimum change support window size")

    args = parser.parse_args()

    if not os.path.isfile(args.file_path):
        raise FileNotFoundError(F"{args.file_path} is not found")
    
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(F"{args.model_path} is not found")
        
    run(args)