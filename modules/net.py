import yaml
import torch
import torch.nn as nn
import torchaudio
from utils import load_yaml
from typing import *


class AudioSegmentationNet(nn.Module):
    def __init__(
            self, 
            num_classes: int,
            sample_rate: int,
            config: Union[str, Dict[str, Any]]="config/config.yaml"
        ):
        super(AudioSegmentationNet, self).__init__()
        if isinstance(config, str):
            self.config = load_yaml(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError(f"config is expected to be str or dict type got {type(config)}")
        
        self.num_classes = num_classes

        self.resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, 
            new_freq=self.config["new_sample_rate"]
        )

        hop_length = n_fft = int(self.config["new_sample_rate"] * (self.config["sample_duration_ms"] / 1000))
        self.config["melspectrogram_config"].update({"hop_length": hop_length, "n_fft": n_fft})
        self.config["mfcc_config"]["melkwargs"].update({"hop_length": hop_length, "n_fft": n_fft})

        self.power_to_db_tfmr = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.melspectogram_tfmr = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config["new_sample_rate"], 
            **self.config["melspectrogram_config"]
        )
        self.mfcc_tfmr = torchaudio.transforms.MFCC(
            sample_rate=self.config["new_sample_rate"], 
            **self.config["mfcc_config"]
        )
        self.register_buffer("taper_window", torch.empty(0), persistent=True)        

        network_config = self.config["network_config"]
        in_features = (4 * self.config["melspectrogram_config"]["n_mels"]) + 2
        self.fc = nn.Sequential(
            nn.Linear(in_features, network_config["hidden_layers_config"]["l1"]),
            nn.Linear(network_config["hidden_layers_config"]["l1"], network_config["hidden_layers_config"]["l2"]),
            nn.Dropout(network_config["dropout"]),
            nn.Linear(network_config["hidden_layers_config"]["l2"], network_config["hidden_layers_config"]["l3"]),
            nn.Dropout(network_config["dropout"]),
            nn.Linear(network_config["hidden_layers_config"]["l3"], self.num_classes),
        )
        self.apply(self.xavier_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input signal (x) size: (N, channels, t_ms * Cf+1 * SR)) 
        # (channels = 1, t_ms = sample duration(ms) Cf = temporal context, SR = Sample Rate)
        assert(x.shape[1] == 1)
        x = self.resampler(x)

        if self.config["taper_input"]:
            # taper the ends of the input signal:
            if self.taper_window.numel() == 0:
                self.taper_window = getattr(torch, f"{self.config['taper_window']}_window")(
                    x.shape[-1], 
                    periodic=False, 
                    device=x.device
                )
            x = x * self.taper_window[None, None, :].tile(1, x.shape[1], 1)
            
        n_frames = (2 * self.config["n_temporal_context"]) + 1
        # ZCR size:               (N, Cf+1)
        # ZCR features size:      (N, 2)
        # mel_spectrogram size:   (N, 1, n_mel, Cf+1)
        # mfcc size:              (N, 1, n_mel, Cf+1)
        # spectral size:          (N, 2, n_mel, Cf+1)
        # spectral_features size: (N, 4*n_mel)
        # features size:          (N, (4*n_mel) + 2) (if n_mel = 20, then size is (N, 82))
        zcr                = AudioSegmentationNet.compute_ZCR(x, n_frames=n_frames).squeeze(dim=1)
        zcr_features       = torch.stack([zcr.mean(dim=1), zcr.std(dim=1)], dim=-1)
        mel_spectrogram    = self.melspectogram_tfmr(x)
        mel_spectrogram    = self.power_to_db_tfmr(mel_spectrogram)
        mel_spectrogram    = AudioSegmentationNet.scale_input(mel_spectrogram)
        mfcc               = self.mfcc_tfmr(x)
        mfcc               = self.power_to_db_tfmr(mfcc)
        mfcc               = AudioSegmentationNet.scale_input(mfcc)
        spectral           = torch.cat((mel_spectrogram, mfcc), dim=1)
        spectral_features  = torch.stack([spectral.mean(dim=3), spectral.std(dim=3)], dim=-1)
        spectral_features  = spectral_features.reshape(spectral_features.shape[0], -1).contiguous()
        features           = torch.cat([spectral_features, zcr_features], dim=1)
        logits             = self.fc(features)
        return logits

    def init_zeros_taper_window(self, taper_window: torch.Tensor):
        self.taper_window = torch.zeros_like(taper_window)

    def xavier_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    @staticmethod
    def scale_input(x: torch.Tensor, e: float=1e-5) -> torch.Tensor:
        # _max = x.max(dim=-1).values.max(dim=-1).values.max(dim=-1).values[:, None, None, None]
        # _min = x.min(dim=-1).values.min(dim=-1).values.min(dim=-1).values[:, None, None, None]
        # return (x - _min) / ((_max - _min) + e)
        mu  = x.mean(dim=(-2, -1))[:, :, None, None]
        std = x.std(dim=(-2, -1))[:, :, None, None]
        return (x - mu) / (std + e)
    
    @staticmethod
    def compute_ZCR(x: torch.Tensor, n_frames: int) -> torch.Tensor:
        # Input shape (x): (N, 1, Cf+1 * t_ms * SR)
        # Cf+1 = n_frames
        assert (x.shape[1] == 1)
        x            = x.reshape(x.shape[0], 1, n_frames, -1)
        sign_changes = x[..., :-1].sign() * x[..., 1:].sign()
        zcr          = (sign_changes < 0).float().sum(3) / x.shape[3]
        return zcr