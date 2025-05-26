import sys
sys.path.append("./xcodec")
import torch
import torch.nn.functional as F
from xcodec.models.soundstream_semantic import SoundStream
from omegaconf import OmegaConf


class XCodecWrapper(SoundStream):
    def __init__(self, config):
        xcodec_config_path = getattr(config, "xcodec_config_path", "./cache/config_hubert_general.yaml")
        xcodec_pretrain_path = getattr(config, "xcodec_pretrain_path", "./cache/xcodec_hubert_general_audio_v2.pt")
        self.xcodec_config = OmegaConf.load(xcodec_config_path)
        super().__init__(**self.xcodec_config.generator.config)
        parameter_dict = torch.load("./cache/xcodec_hubert_general_audio_v2.pth", map_location="cpu")
        self.load_state_dict(parameter_dict)
        self.requires_grad_(False)


    def encode_continuous(self, audio_arr): 
        audio_arr *= self.xcodec_config.audio_norm_scale
        x = audio_arr.unsqueeze(1)

        e_semantic_input = self.get_regress_target(x).detach()
        e_semantic = self.encoder_semantic(e_semantic_input.transpose(1, 2))
        e_acoustic = self.encoder(x)

        if e_acoustic.shape[2] != e_semantic.shape[2]:
            e_acoustic = self.encoder(F.pad(x[:,0,:], (160, 160)).unsqueeze(0)) 
        e= torch.cat([e_acoustic, e_semantic], dim=1)
        e = self.fc_prior(e.transpose(1, 2)).transpose(1, 2)
        return e

    def decode_continuous(self, latent):
        quantized_acoustic = self.fc_post2(latent.transpose(1, 2)).transpose(1, 2)
        return self.decoder_2(quantized_acoustic)