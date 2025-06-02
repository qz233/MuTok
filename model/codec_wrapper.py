import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
codec_path = os.path.join(root_dir, "xcodec")
sys.path.append(codec_path)
import torch
import torch.nn.functional as F
from xcodec.models.soundstream_semantic import SoundStream
from omegaconf import OmegaConf


class XCodecWrapper(SoundStream):
    def __init__(self, config):
        xcodec_config_path = getattr(config, "xcodec_config_path", "./cache/config_hubert_general.yaml")
        xcodec_pretrain_path = getattr(config, "xcodec_pretrain_path", "./cache/xcodec_hubert_general_audio_v2.pt")
        self.norm_factor = getattr(config, "normalize_factor", 1)
        self.xcodec_config = OmegaConf.load(xcodec_config_path)
        super().__init__(**self.xcodec_config.generator.config)
        parameter_dict = torch.load(xcodec_pretrain_path, map_location="cpu")
        self.load_state_dict(parameter_dict)
        self.requires_grad_(False)


    def encode_continuous(self, x): 
        x *= self.xcodec_config.audio_norm_scale

        e_semantic_input = self.get_regress_target(x).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            e_acoustic = self.encoder(F.pad(x[:,0,:], (160, 160)).unsqueeze(0)) 
        e= torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2))   #.transpose(1, 2)  Because we expect (B, L, C)
        e = e / self.norm_factor
        return e

    def decode_continuous(self, latent):
        latent = latent * self.norm_factor
        quantized_acoustic = self.fc_post2(latent).transpose(1, 2)
        return self.decoder_2(quantized_acoustic)
    
    def prepare_input(self, x, y):
        x, y = self.encode_continuous(torch.cat([x, y], dim=0)).chunk(2, dim=0)
        return x, y