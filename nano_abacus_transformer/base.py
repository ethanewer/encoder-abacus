from dataclasses import dataclass
from math import pi, sqrt

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    digit_ids: list[int]
    dropout: float = 0.0
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    n_positions: int = 1024
    use_wpe: bool = True
    vocab_size: int = 50257


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        size_out = x.shape[:-1] + (self.weight.shape[-1],)
        x = torch.addmm(self.bias, x.view(-1, x.shape[-1]), self.weight)
        x = x.view(size_out)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.c_attn = Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = Linear(self.n_embd, self.n_embd)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(self.dropout)

    def forward(self, x: Tensor, is_causal: bool) -> Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        dropout_p = self.dropout if self.training else 0
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=dropout_p, is_causal=is_causal
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    @staticmethod
    def gelu(x: Tensor) -> Tensor:
        return 0.5 * x * (1.0 + torch.tanh(sqrt(2.0 / pi) * (x + 0.044715 * x**3)))

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor, is_causal: bool) -> Tensor:
        x = x + self.attn(self.ln_1(x), is_causal)
        x = x + self.mlp(self.ln_2(x))
        return x


def flat_cross_entropy(logits: Tensor, target: Tensor) -> Tensor:
    return F.cross_entropy(logits.view(-1, logits.shape[-1]), target.view(-1))
