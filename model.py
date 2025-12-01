# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoder(nn.Module):

    def __init__(self, num_freqs: int = 6):
        super().__init__()
        self.num_freqs = num_freqs

    @property
    def out_dim(self) -> int:
        # 3 (원본 xyz) + 3 * 2 * num_freqs (sin/cos)
        return 3 + 3 * 2 * self.num_freqs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., 3]
        return: [..., out_dim]
        """
        out = [x]
        for i in range(self.num_freqs):
            for fn in [torch.sin, torch.cos]:
                out.append(fn((2.0 ** i) * x))
        return torch.cat(out, dim=-1)


class TinyNeRF(nn.Module):
    """아주 작은 NeRF MLP (view direction은 안씀)."""

    def __init__(self, hidden_dim: int = 128, num_freqs: int = 6):
        super().__init__()
        self.pe = PositionalEncoder(num_freqs=num_freqs)
        in_dim = self.pe.out_dim  # 3 + 3*2*num_freqs

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 4)  # (RGB preact, sigma preact)

    def forward(self, x3d: torch.Tensor) -> torch.Tensor:
        """
        x3d: [N, 3] (3D 위치)
        return: [N, 4] (r,g,b, sigma  pre-activation)
        """
        x = self.pe(x3d)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
