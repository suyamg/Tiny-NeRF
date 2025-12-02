# import torch
# from dataset import load_blender_nerf
# from model import TinyNeRF
# from train import train
# from utils import compute_psnr, compute_ssim, compute_lpips
# from config import *
# from utils import get_rays


# TRAIN_LIST = [10, 20, 50, 100, 200, 300]   # 실험할 뷰 수

# def run_experiments():
#     device = torch.device("cuda")

#     for n_view in TRAIN_LIST:
#         print(f"\n========== Training with {n_view} images ==========")

#         images, poses, focal, H, W, testimg, testpose = \
#             load_blender_nerf(BLENDER_SCENE_DIR, device, train_views=n_view)

#         model = TinyNeRF().to(device)
#         optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#         train(
#             model, optimizer, images, poses, focal,
#             H, W, testimg, testpose,
#             SAVE_DIR, N_ITERS, N_SAMPLES, N_RAND, PLOT_STEP,
#             NEAR, FAR, device
#         )

#         # 평가
#         from render import render
#         rays_o, rays_d = get_rays(H, W, focal, testpose)
#         rgb, _ = render(model, rays_o.reshape(-1,3), rays_d.reshape(-1,3),
#                         NEAR, FAR, N_SAMPLES, device)
#         rgb = rgb.reshape(H, W, 3)

#         psnr = compute_psnr(rgb, testimg)
#         ssim = compute_ssim(rgb, testimg)
#         lp = compute_lpips(rgb, testimg)

#         print(f"[{n_view} views] PSNR={psnr:.3f}  SSIM={ssim:.3f}  LPIPS={lp:.3f}")


# if __name__ == "__main__":
#     run_experiments()

import torch
import os
from dataset import load_blender_nerf
from model import TinyNeRF
from train import train
from utils import compute_psnr, compute_ssim, compute_lpips, get_rays
from render import render
from config import *

TRAIN_LIST = [10, 20, 50, 100, 200, 300]

def run_experiments():
    device = torch.device("cuda")

    # 로그 파일 준비
    log_path = os.path.join(SAVE_DIR, "experiment_results.txt")
    with open(log_path, "w") as f:
        f.write("views,psnr,ssim,lpips\n")

    for n_view in TRAIN_LIST:
        print(f"\n========== Training with {n_view} images ==========")

        images, poses, focal, H, W, testimg, testpose = \
            load_blender_nerf(BLENDER_SCENE_DIR, device, train_views=n_view)

        model = TinyNeRF().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        train(
            model, optimizer, images, poses, focal,
            H, W, testimg, testpose,
            SAVE_DIR, N_ITERS, N_SAMPLES, N_RAND, PLOT_STEP,
            NEAR, FAR, device
        )

        # 평가
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render(
            model, 
            rays_o.reshape(-1,3),
            rays_d.reshape(-1,3),
            NEAR, FAR, N_SAMPLES, device
        )

        rgb = rgb.reshape(H, W, 3)

        psnr = compute_psnr(rgb, testimg)
        ssim = compute_ssim(rgb, testimg)
        lp = compute_lpips(rgb, testimg)

        print(f"[{n_view} views] PSNR={psnr:.3f}  SSIM={ssim:.3f}  LPIPS={lp:.3f}")

        # 결과 저장
        with open(log_path, "a") as f:
            f.write(f"{n_view},{psnr:.4f},{ssim:.4f},{lp:.4f}\n")


if __name__ == "__main__":
    run_experiments()
