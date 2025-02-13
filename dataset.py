import os
import logging
import torch
import random
import pandas as pd
import torchaudio
from tqdm import tqdm
from torch.utils.data import IterableDataset, Dataset, get_worker_info
from utils import load_yaml
from typing import *

LOGGER = logging.getLogger(__name__)


class AudioIterableTrainDataset(IterableDataset):
    def __init__(
            self, 
            data_dir: str,
            annotations: Dict[str, Any], 
            config: Union[str, Dict[str, Any]]="config/config.yaml", 
            ext: str="wav",
            ignore_sample_error: bool=False,
            *,
            only_labels: bool=False
        ):
        if isinstance(config, str):
            config = load_yaml(config)
        elif isinstance(config, dict):
            config = config
        else:
            raise ValueError(f"config is expected to be str or dict type got {type(config)}")
        
        self.data_dir             = data_dir
        self.annotations          = annotations
        self.only_labels          = only_labels
        self.ext                  = ext
        _files                    = os.listdir(self.data_dir)
        self.annotations          = {
            k:self.annotations[k] for k in self.annotations.keys() if k+"."+self.ext in _files
        }
        self.files                = list(self.annotations.keys())
        self.sample_duration_ms   = config["sample_duration_ms"]
        self.n_temporal_context   = config["n_temporal_context"]
        temp_file                 = os.path.join(
            self.data_dir, self.files[random.randint(0, len(self.files)-1)] + f".{self.ext}"
        )
        audio_metadata            = torchaudio.info(temp_file)
        self.sample_rate          = audio_metadata.sample_rate
        self.ignore_sample_error = ignore_sample_error

        self._set_classes_details()
    
    def __iter__(self) -> Generator[
        Union[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], Any, None
    ]:
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
            exit_file    = False
            meta_info    = torchaudio.info(
                os.path.join(self.data_dir, files[file_idx] + f".{self.ext}"), backend="soundfile"
            )

            while segment_idx < len(segments):
                exit_segment = False

                for sample in self.split_segments(segments[str(segment_idx)]):
                    if exit_segment:
                        break
                    sample["start"] = max(0, sample["start"])
                    sample["end"]   = min(meta_info.num_frames, sample["end"])
                    class_          = segments[str(segment_idx)]["class"]

                    if self.only_labels:
                        sample = {
                            "file": files[file_idx], **sample, "class": class_, "total_file_frames": meta_info.num_frames
                        }
                        yield sample

                    else:
                        signal, _    = torchaudio.load(
                            os.path.join(self.data_dir, files[file_idx] + f".{self.ext}"),
                            frame_offset=sample["start"],
                            num_frames=sample["end"] - sample["start"],
                            backend="soundfile"
                        )
                        class_       = torch.tensor([self.class2idx[class_]], dtype=torch.int64)
                        timeframe    = torch.tensor([sample["start"], sample["end"]], dtype=torch.float32) / self.sample_rate
                        ndigits      = 4
                        timeframe    = torch.round(timeframe * 10**ndigits) / 10**ndigits # round to (ndigits=4) decimal places
                        segment_size = int(self.sample_duration_ms / 1000 * self.sample_rate)
                        offset       = self.n_temporal_context * segment_size
                        context_size = (offset * 2) + segment_size
                        
                        if signal.shape[1] < context_size:
                            zero_pad = torch.zeros((signal.shape[0], context_size - signal.shape[1], ), dtype=signal.dtype)
                            if sample["start"] == 0:
                                signal = torch.cat([zero_pad, signal], dim=1)
                            elif sample["end"] == meta_info.num_frames:
                                signal = torch.cat([signal, zero_pad], dim=1)
                            else:
                                # This should not occur
                                err_msg = (
                                    f"loaded signal (from {files[file_idx]}) has a size {signal.shape[1]} that is less than"
                                    f" context_size: {context_size} when signal start time ({timeframe[0].item()}) is not 0 and"
                                    f" end time ({timeframe[1].item()}) is not {meta_info.num_frames / self.sample_rate}"
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

    def split_segments(self, current_segment: Dict[str, Any]) -> Generator[Dict[str, Any], Any, None]:
        start        = int(current_segment["start"] * self.sample_rate)
        end          = int(current_segment["end"] * self.sample_rate)
        segment_size = int((self.sample_duration_ms / 1000) * self.sample_rate)
        offset       = self.n_temporal_context * segment_size
        rem          = end % segment_size
        terminal_val = end + ((segment_size - rem) if rem > 0 else 0)
        
        i = start
        while i < terminal_val:
            yield {"start": i - offset, "end": i + offset + segment_size}
            i += segment_size

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


class AudioDataset(Dataset):
    def __init__(
            self, 
            data_dir: str, 
            annotations: Union[str, pd.DataFrame], 
            config: Union[str, Dict[str, Any]],
            ext: str="wav",
            ignore_sample_error: bool=False
        ):
        if isinstance(config, str):
            config = load_yaml(config)
        elif isinstance(config, dict):
            config = config
        else:
            raise ValueError(f"config is expected to be str or dict type got {type(config)}")
        
        if isinstance(annotations, str):
            self.annotations = pd.read_csv(annotations)
        elif isinstance(annotations, pd.DataFrame):
            self.annotations = annotations
        else:
            raise ValueError("annotations is expected to be string path to csv or pandas dataframe")
        
        self.data_dir            = data_dir
        self.sample_duration_ms  = config["sample_duration_ms"]
        self.n_temporal_context  = config["n_temporal_context"]
        self.ext                 = ext
        self.ignore_sample_error = ignore_sample_error
        temp_file                = os.path.join(
            self.data_dir, self.annotations["file"].iloc[random.randint(0, self.annotations.shape[0]-1)] + f".{self.ext}"
        )
        audio_metadata           = torchaudio.info(temp_file)
        self.sample_rate         = audio_metadata.sample_rate

        self._set_classes_details()

    def __len__(self) -> int:
        return self.annotations.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        file, start, end, class_, total_file_frames = self.annotations.iloc[idx, :]

        path         = os.path.join(self.data_dir, f"{file}.{self.ext}")
        signal, _    = torchaudio.load(path, frame_offset=start, num_frames=end-start, backend="soundfile")
        class_       = torch.tensor([self.class2idx[class_]], dtype=torch.int64)
        timeframe    = torch.tensor([start, end], dtype=torch.float32) / self.sample_rate
        ndigits      = 4
        timeframe    = torch.round(timeframe * 10**ndigits) / 10**ndigits # round to (ndigits=4) decimal places
        segment_size = int(self.sample_duration_ms / 1000 * self.sample_rate)
        offset       = self.n_temporal_context * segment_size
        context_size = (offset * 2) + segment_size
        
        if signal.shape[1] < context_size:
            zero_pad = torch.zeros((signal.shape[0], context_size - signal.shape[1], ), dtype=signal.dtype)
            if start == 0:
                signal = torch.cat([zero_pad, signal], dim=1)
            elif end == total_file_frames:
                signal = torch.cat([signal, zero_pad], dim=1)
            else:
                # This should not occur
                err_msg = (
                    f"loaded signal (from {file}) has a size {signal.shape[1]} that is less than"
                    f" context_size: {context_size} when signal start time ({timeframe[0].item()}) is not 0 and"
                    f" end time ({timeframe[1].item()}) is not {total_file_frames / self.sample_rate}"
                )
                if self.ignore_sample_error:
                    LOGGER.warning(err_msg)
                else:
                    raise RuntimeError(err_msg)
                
            if signal.shape[0] > 1:
                signal = signal.mean(dim=0, keepdim=True)

        return signal, class_, timeframe
    
    def get_class_weights(self, device: Optional[Union[str, int]]=None) -> torch.Tensor:
        if not device: device = "cpu"
        label_weights = list(self.class_counts.values())
        label_weights = torch.tensor(label_weights, dtype=torch.float32, device=device)
        label_weights = label_weights.sum() / (label_weights.shape[0] * label_weights)
        return label_weights
    
    def _set_classes_details(self):
        self.classes = set(self.annotations["class"].unique())
        self.classes = sorted(self.classes)
        self.class_counts = {c : (self.annotations["class"] == c).values.sum() for c in self.classes}
        self.class2idx = {}
        self.idx2class = {}
        for i, c in enumerate(self.classes):
            self.class2idx[c] = i
            self.idx2class[i] = c 


class AudioInferenceDataset(IterableDataset):
    def __init__(self, file_path: str, config: Union[str, Dict[str, Any]]):
        if isinstance(config, str):
            config = load_yaml(config)
        elif isinstance(config, dict):
            config = config
        else:
            raise ValueError(f"config is expected to be str or dict type got {type(config)}")
        
        self.file_path          = file_path
        self.audio_metadata     = torchaudio.info(self.file_path)
        self.sample_duration_ms = config["sample_duration_ms"]
        self.n_temporal_context = config["n_temporal_context"]

    def __iter__(self) -> Generator[Tuple[torch.Tensor, torch.Tensor], Any, None]:
        segment_size        = int((self.sample_duration_ms / 1000) * self.audio_metadata.sample_rate)
        offset              = self.n_temporal_context * segment_size
        context_size        = (offset * 2) + segment_size
        start               = 0
        rem                 = self.audio_metadata.num_frames % segment_size
        terminal_val        = self.audio_metadata.num_frames + ((segment_size - rem) if rem > 0 else 0)

        while start < terminal_val:
            adj_start = max(0, start - offset)
            adj_end   = min(self.audio_metadata.num_frames, start + offset + segment_size)

            signal, _ = torchaudio.load(
                self.file_path, 
                frame_offset=adj_start,
                num_frames=adj_end - adj_start, 
                backend="soundfile"
            )
            
            if signal.shape[1] < context_size:
                if adj_start == 0:
                    zero_pad = torch.zeros((1, context_size - signal.shape[1]))
                    signal   = torch.cat([zero_pad, signal], dim=1)

                elif adj_end == self.audio_metadata.num_frames:
                    zero_pad = torch.zeros((1, context_size - signal.shape[1]))
                    signal   = torch.cat([signal, zero_pad], dim=1)

                else:
                    # This should not happen
                    raise RuntimeError(f"signal shape ({signal.shape[1]}) does not match context_size ({context_size})")

            timeframe = torch.tensor([start, min(start + segment_size, self.audio_metadata.num_frames)], dtype=torch.float32)
            timeframe /= self.audio_metadata.sample_rate
            ndigits   = 4
            timeframe = torch.round(timeframe * 10**ndigits) / 10**ndigits
            yield signal, timeframe

            start += segment_size