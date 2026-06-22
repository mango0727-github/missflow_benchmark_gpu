"""
model.py  —  MissFlow v3 (CFMI-style)
======================================
Architecture directly mirrors CFMI (Simkus & Gutmann, 2025):
  - Network input: [t_emb, x_t, x1_cond, cond_mask]
      x_t      = noisy interpolant at TARGET dims, ZERO at cond dims
      x1_cond  = observed values at COND dims, ZERO at target dims
      cond_mask = binary mask indicating which dims are conditioning
  - ResidualFCNetwork with LayerNorm (identical to CFMI's ResidualFCNetwork)
  - 2-layer sinusoidal time embedding (identical to CFMI's TimeEmbeddingNet)
  - Output velocity is multiplied by target_mask externally (in train/sample)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    """2-layer sinusoidal time embedding — identical to CFMI's TimeEmbeddingNet."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) or (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        half = self.dim // 2
        freq = (10.0 ** (torch.arange(half, device=t.device) / max(half - 1, 1) * 4.0))
        table = t * freq.unsqueeze(0)                        # (B, half)
        emb   = torch.cat([table.sin(), table.cos()], dim=1) # (B, dim)
        return F.silu(self.proj2(F.silu(self.proj1(emb))))


class ResBlock(nn.Module):
    """ResidualBlock with LayerNorm — identical to CFMI's ResidualBlock(use_layer_norm=True)."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear1   = nn.Linear(dim, dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.linear2   = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear1(x)
        h = self.layer_norm(h)
        h = F.silu(h)
        h = self.linear2(h)
        return x + h


class VelocityNetwork(nn.Module):
    """
    Velocity field network — mirrors CFMI's VelocityNet + ResidualFCNetwork.

    Input: cat([t_emb, x_t, x1_cond, cond_mask])
      - t_emb    : (B, time_dim)
      - x_t      : (B, d)  interpolant at target dims, zero at cond dims
      - x1_cond  : (B, d)  observed values at cond dims, zero at target dims
      - cond_mask: (B, d)  1 at conditioning dims, 0 at target dims

    Output: (B, d) — raw velocity; caller multiplies by target_mask.
    """
    def __init__(
        self,
        d:          int,
        hidden_dim: int = 256,
        n_layers:   int = 4,
        time_dim:   int = 128,
        # unused kwargs for API compatibility
        use_attention: bool = False,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        assert time_dim % 2 == 0
        input_dim = time_dim + 3 * d   # t_emb + x_t + x1_cond + cond_mask

        self.time_embed  = TimeEmbedding(time_dim)
        self.initial     = nn.Linear(input_dim, hidden_dim)
        self.blocks      = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(n_layers)]
        )
        self.final       = nn.Linear(hidden_dim, d)
        # Zero-init final layer: velocity starts at zero
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def forward(
        self,
        x_t:       torch.Tensor,   # (B, d)  noisy state at target dims
        t:         torch.Tensor,   # (B,)
        x1_cond:   torch.Tensor,   # (B, d)  observed values at cond dims
        cond_mask: torch.Tensor,   # (B, d)  1=conditioning, 0=target
    ) -> torch.Tensor:
        t_emb = self.time_embed(t)                                   # (B, time_dim)
        inp   = torch.cat([t_emb, x_t, x1_cond, cond_mask], dim=-1) # (B, input_dim)
        h     = F.silu(self.initial(inp))
        for block in self.blocks:
            h = F.silu(block(h))
        return self.final(h)
