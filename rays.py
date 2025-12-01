import torch
import numpy as np


def get_rays(H: int, W: int, focal: float,
             pose: torch.Tensor,
             device=None):
    """
    pinhole 카메라에서 각 픽셀에 대한 ray origin, direction 계산.

    pose: [4,4] 또는 [3,4] 카메라 pose (world ← camera)
    return:
        rays_o: [H, W, 3]
        rays_d: [H, W, 3]
    """
    if device is None:
        device = pose.device

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='ij'
    )
    i = i.t()
    j = j.t()

    dirs = torch.stack(
        [(i - W * 0.5) / focal,
         -(j - H * 0.5) / focal,
         -torch.ones_like(i)],
        dim=-1
    )  # [H,W,3]

    c2w = pose[:3, :3]
    rays_d = torch.sum(dirs[..., None, :] * c2w[None, None, :, :], dim=-1)
    rays_o = pose[:3, -1].expand_as(rays_d)
    return rays_o, rays_d


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    cumprod = torch.cumprod(tensor, dim=-1)
    cumprod = torch.roll(cumprod, shifts=1, dims=-1)
    cumprod[..., 0] = 1.0
    return cumprod
