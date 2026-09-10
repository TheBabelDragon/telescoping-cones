from __future__ import annotations

import math

import torch
from torch import nn


class ConeAttention(nn.Module):
    """PyTorch geometric-attention layer.

    Geometry is injected as a bias:

        softmax(QK^T / sqrt(d) + lambda A^cone) V
    """

    def __init__(self, dim: int, lambda_init: float = 1.0, sigma: float = 0.4, aperture: float = 0.5):
        super().__init__()
        self.dim = dim
        self.lambda_param = nn.Parameter(torch.tensor(float(lambda_init)))
        self.sigma = sigma
        self.aperture = aperture
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)

    def cone_geometry(self, positions: torch.Tensor) -> torch.Tensor:
        d2 = ((positions[:, None, :] - positions[None, :, :]) ** 2).sum(-1)
        nrm = torch.nn.functional.normalize(positions, dim=-1)
        cos = (nrm @ nrm.transpose(0, 1)).clamp(-1, 1)
        theta = torch.arccos(cos)
        g = torch.exp(-(theta ** 2) / (2 * self.aperture ** 2))
        return torch.exp(-d2 / (2 * self.sigma ** 2)) * g

    def forward(self, x: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        q, k, v = self.wq(x), self.wk(x), self.wv(x)
        d = q.shape[-1]
        bias = self.cone_geometry(geometry)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d) + self.lambda_param * bias
        w = torch.softmax(scores, dim=-1)
        return w @ v
