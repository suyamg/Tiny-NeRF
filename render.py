# import torch
# import torch.nn.functional as F

# from rays import cumprod_exclusive


# def batchify(fn, chunk=1024 * 32):
#     def ret(inputs):
#         out_list = []
#         for i in range(0, inputs.shape[0], chunk):
#             out_list.append(fn(inputs[i:i + chunk]))
#         return torch.cat(out_list, dim=0)
#     return ret


# def render_rays(model,
#                 rays_o: torch.Tensor,
#                 rays_d: torch.Tensor,
#                 near: float,
#                 far: float,
#                 n_samples: int,
#                 rand: bool = False,
#                 device=None):
#     """
#     Volumetric rendering for a batch of rays.

#     rays_o, rays_d: [N_rays, 3]
#     return:
#         rgb_map:   [N_rays, 3]
#         depth_map: [N_rays]
#         acc_map:   [N_rays]
#     """
#     if device is None:
#         device = rays_o.device

#     N_rays = rays_o.shape[0]

#     # 1. 깊이 샘플링
#     z_vals = torch.linspace(near, far, n_samples, device=device)  # [n_samples]

#     if rand:
#         mids = 0.5 * (z_vals[1:] + z_vals[:-1])
#         upper = torch.cat([mids, z_vals[-1:]], dim=0)
#         lower = torch.cat([z_vals[:1], mids], dim=0)
#         t_rand = torch.rand_like(z_vals)
#         z_vals = lower + (upper - lower) * t_rand

#     # [N_rays, n_samples]
#     z_vals = z_vals.unsqueeze(0).expand(N_rays, n_samples)

#     # 2. 3D 포인트 생성
#     #   rays_o: [N,3] -> [N,1,3]
#     #   rays_d: [N,3] -> [N,1,3]
#     points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
#     flat_points = points.reshape(-1, 3)   # [N_rays*n_samples, 3]

#     # 3. 네트워크 통과
#     raw = batchify(model)(flat_points)    # [...,4]
#     raw = raw.reshape(points.shape[:-1] + (4,))  # [N_rays, n_samples, 4]

#     sigma = F.relu(raw[..., 3])           # [N_rays, n_samples]
#     rgb   = torch.sigmoid(raw[..., :3])   # [N_rays, n_samples, 3]

#     # 4. Volume Rendering
#     one_e_10 = torch.tensor([1e10], dtype=rays_o.dtype, device=device)
#     dists = torch.cat(
#         [z_vals[..., 1:] - z_vals[..., :-1],
#          one_e_10.expand(z_vals[..., :1].shape)],
#         dim=-1
#     )  # [N_rays, n_samples]

#     alpha   = 1.0 - torch.exp(-sigma * dists)
#     weights = alpha * cumprod_exclusive(1.0 - alpha + 1e-10)

#     # RGB, depth, density accumulation
#     rgb_map   = (weights[..., None] * rgb).sum(dim=-2)  # [N_rays,3]
#     depth_map = (weights * z_vals).sum(dim=-1)          # [N_rays]
#     acc_map   = weights.sum(dim=-1)                     # [N_rays]

#     return rgb_map, depth_map, acc_map

import torch
import torch.nn.functional as F
from utils import cumprod_exclusive
from model import positional_encoder

def render(model, rays_o, rays_d, near, far, n_samples, device, chunk=1024):

    # positional encoding L 설정
    if hasattr(model, "module"):
        L_embed = model.module.L_embed
    else:
        L_embed = model.L_embed

    # 샘플링
    z_vals = torch.linspace(near, far, n_samples).to(device)

    N_rays = rays_o.shape[0]

    # 출력 버퍼
    rgb_out   = torch.zeros((N_rays, 3), device=device)
    depth_out = torch.zeros((N_rays,), device=device)
    acc_out   = torch.zeros((N_rays,), device=device)

    for i in range(0, N_rays, chunk):

        rays_o_chunk = rays_o[i:i+chunk]
        rays_d_chunk = rays_d[i:i+chunk]
        N_chunk = rays_o_chunk.shape[0]

        z = z_vals.expand(N_chunk, n_samples)

        pts = rays_o_chunk[..., None, :] + rays_d_chunk[..., None, :] * z[..., :, None]
        pts_flat = pts.reshape(-1, 3)
        # pts_flat = positional_encoder(pts_flat, L_embed)
        raw_list = []
        for j in range(0, pts_flat.shape[0], chunk):
            raw_list.append(model(pts_flat[j:j+chunk]))
        raw = torch.cat(raw_list, dim=0)

        raw = raw.reshape(N_chunk, n_samples, 4)

        rgb   = torch.sigmoid(raw[..., :3])
        sigma = F.relu(raw[..., 3])

        # 거리 계산
        dists = torch.cat([
            z[:, 1:] - z[:, :-1],
            torch.tensor([1e10], device=device).expand(N_chunk, 1)
        ], dim=-1)

        alpha = 1. - torch.exp(-sigma * dists)
        weights = alpha * cumprod_exclusive(1 - alpha + 1e-10)

        # 최종 출력
        rgb_map   = torch.sum(weights[..., None] * rgb, dim=-2)
        depth_map = torch.sum(weights * z, dim=-1)
        acc_map   = torch.sum(weights, dim=-1)

        rgb_out[i:i+chunk]   = rgb_map
        depth_out[i:i+chunk] = depth_map
        acc_out[i:i+chunk]   = acc_map

    return rgb_out, depth_out, acc_out

