"""
Sparse Autoencoder for GPT-2 activations.

See sae_tutorial.md for a conceptual walkthrough of everything here.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class SAEConfig:
    d_model: int = 768          # must match GPT-2's n_emb
    d_sae: int = 768 * 8        # 6144 sparse features
    l1_coeff: float = 5e-3      # sparsity penalty weight


class SparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

        # Decoder bias — also used to center input before encoding
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_model))

        # Encoder: project from d_model to d_sae
        self.W_enc = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_model))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))

        # Decoder: project from d_sae back to d_model
        self.W_dec = nn.Parameter(torch.empty(cfg.d_model, cfg.d_sae))

        # Initialize: W_dec columns are unit-norm, W_enc starts as its transpose
        nn.init.kaiming_uniform_(self.W_dec)
        self.W_dec.data /= self.W_dec.data.norm(dim=0, keepdim=True)
        self.W_enc.data = self.W_dec.data.T.clone()

    def encode(self, x):
        """x: (batch, d_model) -> h: (batch, d_sae)"""
        # Center, project, activate
        x_centered = x - self.b_dec                            # (batch, d_model)
        h = torch.relu(x_centered @ self.W_enc.T + self.b_enc)  # (batch, d_sae)
        return h

    def decode(self, h):
        """h: (batch, d_sae) -> x_hat: (batch, d_model)"""
        return h @ self.W_dec.T + self.b_dec

    def forward(self, x):
        """
        Full forward pass. Returns (loss, mse, l1, h, x_hat).

        x: (batch, d_model) — one activation vector per token position
        """
        h = self.encode(x)          # (batch, d_sae)  — sparse features
        x_hat = self.decode(h)      # (batch, d_model) — reconstruction

        # Losses
        mse = (x - x_hat).pow(2).mean()
        l1 = h.abs().mean()
        loss = mse + self.cfg.l1_coeff * l1

        return loss, mse, l1, h, x_hat
