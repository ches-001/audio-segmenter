import torch
import torch.nn as nn
import torchaudio
from utils import load_yaml
from typing import *


class HarmonicLayer(nn.Linear):
    # PAPER: https://arxiv.org/abs/2502.01628
    
    # NOTE: They say this is an replacement of cross entropy, but it still uses cross entropy
    # The paper is merely an alternative method to computing logits and mutually exclusive 
    # probability scores for classification tasks and the likes, so this is not an replacement
    # of cross entropy, its just an alternative to a Dense Layer + Softmax activation.
    def __init__(
            self, 
            in_features: int, 
            out_features: int, 
            alpha: float=0.01, 
            c: float=1/50304, 
            exp_pow_n: float=1
        ):
        super(HarmonicLayer, self).__init__(in_features, out_features, bias=False)
        self.alpha      = alpha
        self.c          = c
        self.exp_pow_n = exp_pow_n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_embed = pow_n = x.shape[-1]
        pow_n   = pow_n ** self.exp_pow_n
        wx      = torch.einsum("bf, cf -> bc", x, self.weight)
        xx      = torch.norm(x, dim=-1).pow(2)
        ww      = torch.norm(self.weight, dim=-1).pow(2)
        dist_sq = ww + xx[..., None] - (2 * wx)
        dist_sq = dist_sq / n_embed
        dist_sq = dist_sq / dist_sq.min(dim=-1, keepdim=True).values
        dist_sq = dist_sq.pow(-pow_n)
        proba   = dist_sq / dist_sq.sum(dim=-1, keepdim=True)
        proba   = proba + (self.alpha * self.c)
        proba   = proba.clip(min=0.0, max=1.0)
        return proba
        


class AudioSegmentationNet(nn.Module):
    # PAPER: https://aclanthology.org/2016.iwslt-1.4/
    def __init__(
            self, 
            num_classes: int,
            sample_rate: int,
            config: Union[str, Dict[str, Any]]="config/config.yaml",
        ):
        super(AudioSegmentationNet, self).__init__()
        if isinstance(config, str):
            self.config = load_yaml(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise ValueError(f"config is expected to be str or dict type got {type(config)}")
        
        self.num_classes      = num_classes
        self.resampler        = torchaudio.transforms.Resample(
            orig_freq=sample_rate, 
            new_freq=self.config["new_sample_rate"]
        )
        hop_length            = n_fft = int(self.config["new_sample_rate"] * (self.config["sample_duration_ms"] / 1000))
        self.config["mfcc_config"]["melkwargs"].update({"hop_length": hop_length, "n_fft": n_fft})

        self.power_to_db_tfmr = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.mfcc_tfmr        = torchaudio.transforms.MFCC(
            sample_rate=self.config["new_sample_rate"], 
            **self.config["mfcc_config"]
        )
        self.register_buffer("taper_window", torch.empty(0), persistent=True)        

        network_config = self.config["network_config"]
        in_features    = (3 * self.config["mfcc_config"]["melkwargs"]["n_mels"]) + 3
        self.hidden_fc = nn.Sequential(
            nn.Linear(in_features, network_config["hidden_layers_config"]["l1"]),
            nn.Sigmoid(),

            nn.Linear(network_config["hidden_layers_config"]["l1"], network_config["hidden_layers_config"]["l2"]),
            nn.Sigmoid(),

            nn.Linear(network_config["hidden_layers_config"]["l2"], network_config["hidden_layers_config"]["l3"]),
            nn.Sigmoid(),
        )
        if not network_config["use_harmonic_layer"]:
            self.final_fc = nn.Sequential(
                nn.Linear(network_config["hidden_layers_config"]["l3"], self.num_classes),
                nn.Softmax(dim=-1)
            )
        else:
            self.final_fc = HarmonicLayer(
                network_config["hidden_layers_config"]["l3"], 
                self.num_classes, 
                **network_config["harmonic_layer_config"]
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
            
        context_size = (2 * self.config["n_temporal_context"]) + 1
        # ZCR size:               (N, Cf+1)
        # ZCR features size:      (N, 3)
        # mfcc size:              (N, 1, n_mel, Cf+1)
        # spectral_features size: (N, 3*n_mel)
        # features size:          (N, (3*n_mel) + 3) (if n_mel = 20, then size is (N, 63))
        zcr                = AudioSegmentationNet.compute_ZCR(x, context_size=context_size).squeeze(dim=1)
        zcr_features       = torch.stack([zcr.mean(dim=1), zcr.std(dim=1), zcr.std(dim=1).pow(2)], dim=-1)
        mfcc               = self.mfcc_tfmr(x)
        mfcc               = self.power_to_db_tfmr(mfcc)
        mfcc               = self.scale_input(mfcc)
        spectral_features  = torch.stack([mfcc.mean(dim=3), mfcc.std(dim=3), mfcc.std(dim=3).pow(2)], dim=-1)
        spectral_features  = spectral_features.reshape(spectral_features.shape[0], -1)
        features           = torch.cat([spectral_features, zcr_features], dim=1).contiguous()
        features           = self.hidden_fc(features)
        proba              = self.final_fc(features)
        return proba

    def init_zeros_taper_window(self, taper_window: torch.Tensor):
        self.taper_window = torch.zeros_like(taper_window)

    def xavier_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if torch.is_tensor(m.bias):
                m.bias.data.fill_(0.01)

    def scale_input(self, x: torch.Tensor, e: float=1e-7) -> torch.Tensor:
        _max = x.max(dim=-1).values.max(dim=-1).values.max(dim=-1).values[:, None, None, None]
        _min = x.min(dim=-1).values.min(dim=-1).values.min(dim=-1).values[:, None, None, None]
        return (x - _min) / ((_max - _min) + e)

    @staticmethod
    def compute_ZCR(x: torch.Tensor, context_size: int) -> torch.Tensor:
        # Input shape (x): (N, 1, Cf+1 * t_ms * SR)
        # Cf+1 = context_size
        assert (x.shape[1] == 1)
        remainder = x.shape[-1] % context_size 
        if remainder == 0:
            x = x.reshape(x.shape[0], 1, context_size, -1)
        else:
            x = x[..., :x.shape[-1]-remainder].reshape(x.shape[0], 1, context_size, -1)
        sign_changes = x[..., :-1].sign() * x[..., 1:].sign()
        zcr          = (sign_changes < 0).float().sum(3) / x.shape[3]
        return zcr