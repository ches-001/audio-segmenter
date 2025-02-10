import os
import math
import logging
import torch
import random
import torchaudio
from tqdm import tqdm
from torch.utils.data import IterableDataset, get_worker_info
from utils import load_yaml
from typing import *

LOGGER = logging.getLogger(__name__)

class AudioTrainDataset(IterableDataset):
    def __init__(
            self, 
            data_dir: str,
            annotations: Dict[str, Any], 
            config: Union[str, Dict[str, Any]]="config/config.yaml", 
            ext: str="wav",
            ignore_sample_error: bool=False
        ):
        if isinstance(config, str):
            config = load_yaml(config)
        elif isinstance(config, dict):
            config = config
        else:
            raise ValueError(f"config is expected to be str or dict type got {type(config)}")
        
        self.data_dir             = data_dir
        self.annotations          = annotations
        self.ext                  = ext
        _files                    = os.listdir(self.data_dir)
        self.annotations          = {k:self.annotations[k] for k in self.annotations.keys() if k+"."+self.ext in _files}
        self.files                = list(self.annotations.keys())
        self.sample_duration_ms   = config["sample_duration_ms"]
        self.n_temporal_context   = config["n_temporal_context"]
        temp_file                 = os.path.join(self.data_dir, self.files[random.randint(0, len(self.files)-1)] + f".{self.ext}")
        audio_metadata            = torchaudio.info(temp_file)
        self.sample_rate          = audio_metadata.sample_rate
        self.ignore_sample_error = ignore_sample_error

        self._set_classes_details()

    
    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Any, None]:
        worker_info = get_worker_info()

        if not worker_info:
            files = self.files
        else:
            n_workers = worker_info.num_workers
            job_size  = len(self.files) // n_workers
            if job_size == 0:
                if worker_info.id > len(self.files) - 1:
                    files = []
                else:
                    files = self.files
            else:
                job_start = job_size * worker_info.id
                if worker_info.id < worker_info.num_workers - 1:
                    files = self.files[job_start : job_start + job_size]
                else:
                    files =  self.files[job_start : ]
            
        random.shuffle(files)
        file_idx = 0
                
        while file_idx < len(files):
            segment_idx  = 0
            segments     = self.annotations[self.files[file_idx]]
            segment_keys = list(segments.keys())
            exit_file    = False

            while segment_idx < len(segments):
                exit_segment = False

                for sample in self.split_segments(segments[str(segment_idx)]):
                    if exit_segment:
                        break

                    signal, _ = torchaudio.load(
                        os.path.join(self.data_dir, files[file_idx] + f".{self.ext}"),
                        frame_offset=math.floor(sample["start"] * self.sample_rate),
                        num_frames=math.ceil((sample["end"] - sample["start"]) * self.sample_rate),
                        backend="soundfile"
                    )
                    class_ = torch.tensor([self.class2idx[sample["class"]]], dtype=torch.int64)
                    timeframe = torch.tensor([sample["start"], sample["end"]], dtype=torch.float32)

                    context_size = ((self.n_temporal_context * 2) + 1) * self.sample_duration_ms / 1000 * self.sample_rate
                    context_size = math.ceil(context_size)
                    
                    if signal.shape[1] < context_size:
                        zero_pad = torch.zeros((signal.shape[0], context_size - signal.shape[1], ), dtype=signal.dtype)
                        if sample["start"] == 0:
                            signal = torch.cat([zero_pad, signal], dim=1)
                        elif sample["end"] > segments[segment_keys[-1]]["end"]:
                            signal = torch.cat([signal, zero_pad], dim=1)
                        else:
                            # This should not occur
                            err_msg = (
                                f"loaded signal (from {files[file_idx]}) has a size {signal.shape[1]} that is less than"
                                f" context_size: {context_size} when signal start time ({sample['start']}) is not 0 and"
                                f" end time ({sample['end']}) is not {segments[segment_keys[-1]]['end']}"
                            )
                            if self.ignore_sample_error:
                                LOGGER.warning(err_msg)
                                exit_segment = True
                                exit_file    = True
                                continue
                            else:
                                raise RuntimeError(err_msg)
                    
                    if signal.shape[0] > 1:
                        signal = signal.mean(dim=0, keepdim=True)

                    yield signal, class_, timeframe

                if exit_file:
                    break
                segment_idx += 1

            file_idx += 1


    def split_segments(self,current_segment: Dict[str, Any]) -> Generator[Dict[str, Any], Any, None]:
        start     = current_segment["start"]
        end       = current_segment["end"]
        class_    = current_segment["class"]
        increment = self.sample_duration_ms / 1000
        offset    = self.n_temporal_context * increment
        
        i = start
        while i < (end - increment):
            yield {
                "start": round(max(0, i - offset), 4), 
                "end": round(i + offset + increment, 4),
                "class": class_
            }
            i += increment
            i = round(i, 4)


    def get_class_weights(self, device: Optional[Union[str, int]]=None) -> torch.Tensor:
        if not device: device = "cpu"
        label_weights = list(self.class_counts.values())
        label_weights = torch.tensor(label_weights, dtype=torch.float32, device=device)
        label_weights = label_weights.sum() / (label_weights.shape[0] * label_weights)
        return label_weights
    
    def _set_classes_details(self):
        self.classes = set()
        self.class_counts = {}
        for file in tqdm(self.annotations.keys()):
            for segment in self.annotations[file]:
                class_ = self.annotations[file][segment]["class"]
                self.classes.add(class_)
                if class_ not in self.class_counts:
                    self.class_counts[class_] = 1
                else:
                    self.class_counts[class_] += 1
        
        self.classes = sorted(self.classes)
        self.class_counts = {c : self.class_counts[c] for c in self.classes}
        self.class2idx = {}
        self.idx2class = {}

        for i, c in enumerate(self.classes):
            self.class2idx[c] = i
            self.idx2class[i] = c        


class AudioInferenceDataset(IterableDataset):
    def __init__(self, file_path: str, config: Union[str, Dict[str, Any]]):
        self.file_path          = file_path
        self.audio_metadata     = torchaudio.info(self.file_path)
        config                  = load_yaml(config) if isinstance(config, str) else config
        self.sample_duration_ms = config["sample_duration_ms"]
        self.n_temporal_context = config["n_temporal_context"]

    
    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], Any, None]:
        audio_duration      = self.audio_metadata.num_frames / self.audio_metadata.sample_rate
        segment_duration    = self.sample_duration_ms / 1000
        offset_duration     = self.n_temporal_context * segment_duration
        context_duration    = (offset_duration * 2) + segment_duration
        context_size        = math.ceil(context_duration * self.audio_metadata.sample_rate)
        start               = 0
        increment           = self.sample_duration_ms / 1000

        while start < (audio_duration - increment):
            adj_start = round(max(0, start - offset_duration), 4)
            adj_end   = round(start + offset_duration + segment_duration, 4)

            signal, _ = torchaudio.load(
                self.file_path, 
                frame_offset=math.floor(adj_start * self.audio_metadata.sample_rate), 
                num_frames=math.ceil((adj_end - adj_start) * self.audio_metadata.sample_rate), 
                backend="soundfile"
            )
            
            if signal.shape[1] < context_size:
                if adj_start == 0:
                    zero_pad = torch.zeros((1, context_size - signal.shape[1]))
                    signal   = torch.cat([zero_pad, signal], dim=1)

                elif adj_end > audio_duration:
                    zero_pad = torch.zeros((1, context_size - signal.shape[1]))
                    signal   = torch.cat([signal, zero_pad], dim=1)

                else:
                    # This should not happen
                    raise RuntimeError(f"signal shape ({signal.shape[1]}) does not match context_size ({context_size})")

            timeframe = torch.tensor([start, min(start + (self.sample_duration_ms / 1000), audio_duration)], dtype=torch.float32)
            yield signal, timeframe

            start += increment
            start = round(start, 4)