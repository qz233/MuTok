#ref: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from diffusers.models.embeddings import SinusoidalPositionalEmbedding, Timesteps, TimestepEmbedding


class DecoderAdaLN(nn.Module):
    # Add the conditional information and ts though adaptive layer norm.
    def __init__(self, hidden_dim):
        super().__init__()
        self.timestep = Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=hidden_dim)
        self.register_embedding = nn.Embedding(1, hidden_dim)
        self.condition_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.norm = nn.LayerNorm(hidden_dim)
    def forward(self, x, timestep, condition):
        num_registers = x.shape[1] - condition.shape[1]
        B = x.shape[0]
        # timestep
        ts = self.timestep(timestep)
        ts = self.timestep_embedder(ts)
        # condition
        condition = self.condition_proj(condition)
        # register
        reg_embedding = self.register_embedding.weight.unsqueeze(0).expand((B, num_registers, -1))
        condition = torch.cat([condition, reg_embedding], dim=1)
        emb = self.norm_proj(F.silu(ts.unsqueeze(1) + condition))
        # ada norm
        scale, shift = emb.chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale) + shift
        return x 

class Attention(nn.Module):
    def __init__(self, num_embedding, num_heads, dropout_p=0.0, is_casual=False):
        super().__init__()
        self.c_attn = nn.Linear(num_embedding, 3 * num_embedding)
        self.c_proj = nn.Linear(num_embedding, num_embedding)
        # regularization
        self.n_head = num_heads
        self.n_embd = num_embedding
        self.is_casual = is_casual
        # dropout
        self.dropout_rate = dropout_p
        self.resid_dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, mask=None):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v,
                                                              attn_mask=mask, 
                                                              dropout_p=self.dropout_rate if self.training else 0,
                                                              is_causal=self.is_casual)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, num_embedding, dropout_p=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(num_embedding, 4 * num_embedding)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * num_embedding, num_embedding)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class AttentionBlock(nn.Module):
    def __init__(self, num_embedding, num_heads, norm=nn.LayerNorm, dropout_p=0.0, is_casual=False):
        super().__init__()
        self.ln_1 = norm(num_embedding)
        self.attn = Attention(num_embedding, num_heads, is_casual=is_casual, dropout_p=dropout_p)
        self.ln_2 = norm(num_embedding)
        self.mlp = MLP(num_embedding, dropout_p=dropout_p)

    def forward(self, x, ts=None, condition=None):
        if ts is None:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            # add timestep and condition though Adaptive norm.
            x = x + self.attn(self.ln_1(x, ts, condition))
            x = x + self.mlp(self.ln_2(x, ts, condition))
        return x

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len

        self.pe = SinusoidalPositionalEmbedding(config.num_embedding, self.seq_len)
        self.register_tokens = nn.Embedding(config.num_register_tokens, config.num_embedding)
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(config.num_embedding, config.num_heads, dropout_p=config.dropout_p) 
            for _ in range(config.encoder_num_layers)
        ])
        self.in_proj = nn.Linear(config.input_dim, config.num_embedding)
    def forward(self, x):
        x = self.in_proj(x)
        B, L, _ = x.shape
        x = self.pe(x)
        # apply pe before concat registers, because they are global tokens.
        registers = self.register_tokens.weight.expand((B , -1, -1))
        x = torch.cat((x, registers), dim=1)
        for block in self.attention_blocks:
            x = block(x)
        # seperately return music tokens and register tokens
        return x[:, :L], x[:, L:]


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len
        self.token_len = config.seq_len // config.codec_sample_rate
        
        self.pe = SinusoidalPositionalEmbedding(config.num_embedding, self.seq_len)

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(config.num_embedding, config.num_heads, norm=DecoderAdaLN, dropout_p=config.dropout_p)
            for _ in range(config.decoder_num_layers)
        ])
        self.in_proj = nn.Linear(config.input_dim, config.num_embedding)
        self.out_proj = nn.Linear(config.num_embedding, config.input_dim)
    def forward(self, x, ts, condition):
        # expand latent tokens
        condition, registers = condition[:, :self.token_len], condition[:, self.token_len:]
        condition = condition.unsqueeze(2).expand((-1, -1, self.config.codec_sample_rate, -1))
        condition = condition.flatten(1, 2)

        x = self.in_proj(x)
        x = self.pe(x)
        x = torch.cat((x, registers), dim=1)
        for block in self.attention_blocks:
            x = block(x, ts, condition)
        x = self.out_proj(x)
        return x[:, :self.seq_len]


