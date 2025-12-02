import torch
from piq import ssim
import lpips


# ---------------------------
# Evaluation Metrics
# ---------------------------
def compute_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    psnr = -10 * torch.log10(mse)
    return psnr.item()


lpips_fn = lpips.LPIPS(net='alex')
def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    """
    Computes exclusive cumulative product
    Example:
        input:  [a, b, c]
        output: [1, a, a*b]
    """
    cumprod = torch.cumprod(tensor, dim=-1)
    cumprod = torch.roll(cumprod, shifts=1, dims=-1)
    cumprod[..., 0] = 1.0
    return cumprod

def compute_lpips(pred, gt):
    pred_n = pred.permute(2,0,1).unsqueeze(0)*2-1
    gt_n = gt.permute(2,0,1).unsqueeze(0)*2-1
    return lpips_fn(pred_n, gt_n).item()


def compute_ssim(pred, gt):
    pred_n = pred.permute(2,0,1).unsqueeze(0)
    gt_n = gt.permute(2,0,1).unsqueeze(0)
    return ssim(pred_n, gt_n, data_range=1.0).item()


# ---------------------------
# Data augmentation
# ---------------------------
def augment_image(img):
    img = img * (0.9 + 0.2 * torch.rand(1, device=img.device))
    img += 0.02 * torch.randn_like(img)
    return torch.clamp(img, 0, 1)


def jitter_rays(rays_o, rays_d):
    rays_o = rays_o + 0.01 * torch.randn_like(rays_o)
    rays_d = rays_d + 0.01 * torch.randn_like(rays_d)
    return rays_o, rays_d


def get_rays(H, W, focal, c2w):
    """
    Generate camera rays for each pixel.
    """
    device = c2w.device  # ← pose 가 있는 device 로 자동 설정

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='ij'
    )

    dirs = torch.stack(
        [(i - W * 0.5) / focal,
         -(j - H * 0.5) / focal,
         -torch.ones_like(i, device=device)],     # ← device 지정
        dim=-1
    )

    # 회전 적용 (CUDA 상에서 수행)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)

    # 레이 원점 (CUDA 상)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d
