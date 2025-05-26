import torch
import torch.nn as nn
import torch.nn.functional as F

from quantizer import SimVQ1D
from transformer import Encoder, Decoder

from minrf import RF


class Mutok(nn.Module):
    def __init__(self, config):
        self.config = config
        self.seq_len = config.seq_len
        self.codec_sample_rate = config.codec_sample_rate
        self.cfg_dropout_rate = getattr(config, "cfg_dropout_rate", 0.1)

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quantizer = SimVQ1D(config.num_embedding, 8)
        self.rf_decoder = RF(self.decoder, ln=True)

        self.in_proj = nn.Linear(config.input_dim, config.num_embedding)
        self.out_proj = nn.Linear(config.num_embedding, config.output_dim)
    def forward(self, x):
        # in the forward pass, we reconstrunct the input and calculate 
        # the reconstruction loss as well as other losses.
        x = self.in_proj(x)
        B, L, C = x.shape
        z, register = self.encoder(x)

        # average every second's feature, and quantize it
        z = z.reshape((B, L // self.codec_sample_rate, self.codec_sample_rate, C)).mean(dim=2)
        z = torch.cat((z, register), dim=1)
        (z_q, _, indices), vq_loss = self.quantizer(z)

        # classifer-free guidance training
        if torch.random.rand() < self.cf_prob:
            z_q = torch.zeros_like(z_q)

        # denoise training
        reconstruction_loss, loss_info = self.rf_decoder(x, z_q)
        return reconstruction_loss + vq_loss

    def inference(self, tokens):
        z_q = self.quantizer.get_codebook_entry(tokens, shape=None)
        null_condition = torch.zeros_like(z_q)
        B, L, C = z_q.shape
        noise = torch.randn((B, L * self.codec_sample_rate, C)).to(z_q.device)

        return self.rf_decoder.inference(noise, z_q, 
                                         null_cond=null_condition, 
                                         cfg=getattr(self.config, 'cfg', 2.0))
    
    def reconstruction(self, x):
        x = self.in_proj(x)
        B, L, C = x.shape
        z, register = self.encoder(x)

        # average every second's feature, and quantize it
        z = z.reshape((B, L // self.codec_sample_rate, self.codec_sample_rate, C)).mean(dim=2)
        z = torch.cat((z, register), dim=1)
        (z_q, _, indices), vq_loss = self.quantizer(z)

        return self.inference(indices)
    