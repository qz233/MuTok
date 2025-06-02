import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import SimVQ1D
from .transformer import Encoder, MutokDecoder
from .minrf import RF


class Mutok(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.codec_sample_rate = config.codec_sample_rate
        self.cfg_dropout_rate = getattr(config, "cfg_dropout_rate", 0.1)

        self.encoder = Encoder(config)
        self.decoder = MutokDecoder(config)
        self.quantizer = SimVQ1D(config.codebook_size, config.num_embedding)
        self.rf_decoder = RF(self.decoder, ln=True)

    def forward(self, x):
        # x[1500, 512]
        # in the forward pass, we reconstrunct the input and calculate 
        # the reconstruction loss as well as other losses.
        B, L, _ = x.shape
        z, register = self.encoder(x)
        # z[1500, 512]
        # average every second music feature, and quantize it
        z = z.reshape((B, L // self.codec_sample_rate, self.codec_sample_rate, self.config.num_embedding)).mean(dim=2)
        z = torch.cat((z, register), dim=1)
        (z_q, _, indices), vq_loss = self.quantizer(z)

        # classifer-free guidance training
        if torch.rand((1,)).item() < self.cfg_dropout_rate:
            z_q = torch.zeros_like(z_q)

        # denoise
        reconstruction_loss, loss_info = self.rf_decoder(x, z_q)
        print(loss_info, vq_loss.commitment)
        return reconstruction_loss + vq_loss.commitment

    def inference(self, tokens):
        z_q = self.quantizer.get_codebook_entry(tokens, shape=None)
        null_condition = torch.zeros_like(z_q)
        B, L, C = z_q.shape
        noise = torch.randn((B, L * self.codec_sample_rate, C)).to(z_q.device)
        # rectified flow ode sampling
        return self.rf_decoder.inference(noise, z_q, 
                                         null_cond=null_condition, 
                                         cfg=getattr(self.config, 'cfg', 2.0))
    
    def encode(self, x):
        B, L, _ = x.shape
        z, register = self.encoder(x)

        # average every second music feature, and quantize it
        z = z.reshape((B, L // self.codec_sample_rate, self.codec_sample_rate, self.config.num_embedding)).mean(dim=2)
        z = torch.cat((z, register), dim=1)
        (z_q, _, indices), vq_loss = self.quantizer(z)
        return indices
    def reconstruction(self, x):
        indices = self.encode(x)
        return self.inference(indices)
    