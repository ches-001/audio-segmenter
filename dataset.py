import os
import math
import torch
import random
import torchaudio
from tqdm import tqdm
from torch.utils.data import IterableDataset, get_worker_info
from utils import load_yaml
from typing import *

class AudioTrainDataset(IterableDataset):
    def __init__(
            self, 
            data_dir: str,
            annotations: Dict[str, Any], 
            config: Union[str, Dict[str, Any]]="config/config.yaml", 
            ext: str="wav"
        ):
        if isinstance(config, str):
            config = load_yaml(config)
        elif isinstance(config, dict):
            config = config
        else:
            raise ValueError(f"config is expected to be str or dict type got {type(config)}")
        
        self.data_dir           = data_dir
        self.annotations        = annotations
        self.ext                = ext
        _files                  = os.listdir(self.data_dir)
        self.annotations        = {k:self.annotations[k] for k in self.annotations.keys() if k+"."+self.ext in _files}
        self.files              = list(self.annotations.keys())
        self.sample_duration_ms = config["sample_duration_ms"]
        self.n_temporal_context = config["n_temporal_context"]
        temp_file               = os.path.join(self.data_dir, self.files[random.randint(0, len(self.files))] + f".{self.ext}")
        audio_metadata          = torchaudio.info(temp_file)
        self.sample_rate        = audio_metadata.sample_rate

        self._set_classes_details()

    
    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Any, None]:
        worker_info = get_worker_info()

        if not worker_info:
            files = self.files
        else:
            n_workers   = worker_info.num_workers
            job_size  = len(self.files) // n_workers
            if job_size == 0:
                if worker_info.id > len(self.files) - 1:
                    files = []
                else:
                    files = self.files
            else:
                job_start = job_size * worker_info.id
                files = self.files[job_start : job_start + job_size]
            
        file_idx = 0
                
        while file_idx < len(files):
            segment_idx = 0
            segments    = self.annotations[self.files[file_idx]]
            
            while segment_idx < len(segments):
                for sample in self.split_segments(
                    files[file_idx], 
                    segments[str(segment_idx)],
                    segments[str(segment_idx-1)] if segment_idx != 0 else None,
                    segments[str(segment_idx+1)] if segment_idx+1 < len(segments) else None,
                    self.sample_duration_ms,
                    self.n_temporal_context
                ):
                    signal, _ = torchaudio.load(
                        os.path.join(self.data_dir, files[file_idx] + f".{self.ext}"),
                        frame_offset=int(sample["start"] * self.sample_rate),
                        num_frames=int((sample["end"] - sample["start"]) * self.sample_rate),
                        backend="soundfile"
                    )
                    class_ = torch.tensor([self.class2idx[sample["class"]]], dtype=torch.int64)
                    timeframe = torch.tensor([sample["start"], sample["end"]], dtype=torch.float32)

                    t_size = ((self.n_temporal_context * 2) + 1) * self.sample_rate * self.sample_duration_ms / 1000
                    t_size = math.floor(t_size)
                    
                    if signal.shape[1] < t_size:
                        if signal.shape[1] > (t_size // 2):
                            zero_pad = torch.zeros((signal.shape[0], t_size - signal.shape[1], ), dtype=signal.dtype)
                            signal   = torch.cat([signal, zero_pad], dim=1)
                        else:
                            continue
                    
                    if signal.shape[0] > 1:
                        signal = signal.mean(dim=0, keepdim=True)
                    yield signal, class_, timeframe

                segment_idx += 1

            file_idx += 1


    def split_segments(
        self,
        filename: str,
        current_segment: Dict[str, Any], 
        prev_segment: Optional[Dict[str, Any]], 
        next_segment: Optional[Dict[str, Any]],
        duration_ms: float=10,
        temporal_context: int=6
    ) -> Generator[Dict[str, Any], Any, None]:
        
        start = current_segment["start"]
        end = current_segment["end"]
        class_ = current_segment["class"]
        offset = (temporal_context * duration_ms) / 1000

        if prev_segment is not None and start != 0:
            start -= offset

        if next_segment is not None:
            end += offset

        sample_duration = (((2 * temporal_context) + 1) * duration_ms) / 1000

        i = start
        while i < end:
            yield {
                "filename": filename, 
                "start": round(i, 4), 
                "end": round(i + sample_duration, 4), 
                "class": class_
            }
            i += (duration_ms / 1000)


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
        segment_size        = (self.sample_duration_ms / 1000) * self.audio_metadata.sample_rate
        offset              = self.n_temporal_context * segment_size
        context_size        = int((offset * 2) + segment_size)
        start               = 0

        while start < audio_duration:
            adj_start = int(start * self.audio_metadata.sample_rate - offset)
            adj_end   = int(start * self.audio_metadata.sample_rate + offset + segment_size)
            signal, _ = torchaudio.load(
                self.file_path, 
                frame_offset=max(adj_start, 0), 
                num_frames=adj_end - adj_start, 
                backend="soundfile"
            )
            
            if adj_start < 0:
                zero_pad = torch.zeros((1, context_size - signal.shape[1]))
                signal   = torch.cat([zero_pad, signal], dim=1)

            if adj_end > self.audio_metadata.num_frames:
                zero_pad = torch.zeros((1, context_size - signal.shape[1]))
                signal   = torch.cat([signal, zero_pad], dim=1)

            timeframe = torch.tensor([start, min(start + self.sample_duration_ms / 1000, audio_duration)], dtype=torch.float32)
            yield signal, timeframe

            start += self.sample_duration_ms / 1000