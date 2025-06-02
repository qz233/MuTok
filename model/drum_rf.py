import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import Encoder, SimpleDecoder
from .minrf import RF

class DrumRFDiT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.codec_sample_rate = config.codec_sample_rate
        self.cfg_dropout_rate = getattr(config, "cfg_dropout_rate", 0.1)

        self.condition_encoder = Encoder(config, use_register=False)
        self.decoder = SimpleDecoder(config)
        self.rf_decoder = RF(self.decoder, ln=True)

    def forward(self, x, cond):
        # x[B, L, D_in], cond[B, L, D_in]
        # process condition
        cond = self.condition_encoder(cond)
        
        # classifer-free guidance training
        if torch.rand((1,)).item() < self.cfg_dropout_rate:
            cond = torch.zeros_like(cond)

        # denoise (condition though AdaNorm)
        reconstruction_loss, loss_info = self.rf_decoder(x, cond)
        return reconstruction_loss
    
    def inference(self, cond):
        noise = torch.randn_like(cond)
        cond = self.condition_encoder(cond)
        null_cond = torch.zeros_like(cond)
        # rectified flow ode sampling
        return self.rf_decoder.sample(noise, cond, 
                                         null_cond=null_cond, 
                                         cfg=getattr(self.config, 'cfg', 2.0))
