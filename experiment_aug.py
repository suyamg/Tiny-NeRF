import torch
import os
from dataset import load_blender_nerf
from model import TinyNeRF
from train import train, TrainConfig
from render import render
from utils import compute_psnr, compute_ssim, compute_lpips, get_rays
from config import *

def run_exp(use_aug, tag):
    print(f"\n===== Augmentation {tag} =====")

    # Aug 설정
    TrainConfig.USE_AUGMENT = use_aug

    device = torch.device("cuda")

    images, poses, focal, H, W, testimg, testpose = \
        load_blender_nerf(BLENDER_SCENE_DIR, device, train_views=100)

    model = TinyNeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(
        model, optimizer, images, poses, focal,
        H, W, testimg, testpose,
        SAVE_DIR, N_ITERS, N_SAMPLES, N_RAND, PLOT_STEP,
        NEAR, FAR, device
    )

    # 평가
    model.eval()
    with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, focal, testpose)
        rgb, depth, acc = render(
            model,
            rays_o.reshape(-1,3),
            rays_d.reshape(-1,3),
            NEAR, FAR, N_SAMPLES, device,
            chunk=1024
        )
        rgb = rgb.reshape(H, W, 3)

        psnr = compute_psnr(rgb, testimg)
        ssim = compute_ssim(rgb, testimg)
        lp = compute_lpips(rgb, testimg)

    print(f"[{tag}] PSNR={psnr:.3f}, SSIM={ssim:.3f}, LPIPS={lp:.3f}")

    return psnr, ssim, lp


if __name__ == "__main__":
    no_aug = run_exp(False, "NO AUG")
    yes_aug = run_exp(True, "WITH AUG")

    print("\n======== RESULT SUMMARY ========")
    print(" NO AUG    :", no_aug)
    print(" WITH AUG  :", yes_aug)

    log = open(os.path.join(SAVE_DIR, "augmentation_result.txt"), "w")
    log.write(f"NO_AUG,{no_aug}\n")
    log.write(f"WITH_AUG,{yes_aug}\n")
    log.close()
