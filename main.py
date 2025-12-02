# import torch
# import torch.nn as nn


# from config import (
#     DATA_PATH, SAVE_DIR,
#     N_ITERS, N_SAMPLES, N_RAND,
#     PLOT_STEP, LR, NEAR, FAR, SEED,
#     DATASET_TYPE, BLENDER_SCENE_DIR,  # 이거도 있으면 같이
# )
# from dataset import load_tiny_nerf, load_blender_nerf
# from model import TinyNeRF
# from train import train



# def main():
#     # ===== Device & Seed =====
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Device:", device)

#     if SEED is not None:
#         import random
#         import numpy as np
#         random.seed(SEED)
#         np.random.seed(SEED)
#         torch.manual_seed(SEED)
#         if device.type == "cuda":
#             torch.cuda.manual_seed_all(SEED)

#     # ===== 데이터 로딩 =====
#     if DATASET_TYPE == "tiny":
#         print(">> Using tiny_nerf_data.npz")
#         images, poses, focal, H, W, testimg, testpose = load_tiny_nerf(DATA_PATH, device)
#     elif DATASET_TYPE == "blender":
#         print(">> Using NeRF Synthetic (Blender) dataset")
#         images, poses, focal, H, W, testimg, testpose = load_blender_nerf(BLENDER_SCENE_DIR, device)
#     else:
#         raise ValueError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")

#     print("Loaded data:", images.shape, poses.shape, "focal:", focal)

#     # ===== 모델 & 옵티마이저 =====
#     nerf = TinyNeRF()
#     nerf = nn.DataParallel(nerf).to(device)

#     optimizer = torch.optim.Adam(nerf.parameters(), lr=LR, eps=1e-5)

#     # ===== 학습 =====
#     train(
#         model=nerf,
#         optimizer=optimizer,
#         images=images,
#         poses=poses,
#         focal=focal,
#         H=H,
#         W=W,
#         testimg=testimg,
#         testpose=testpose,
#         save_dir=SAVE_DIR,
#         n_iters=N_ITERS,
#         n_samples=N_SAMPLES,
#         N_rand=N_RAND,
#         plot_step=PLOT_STEP,
#         near=NEAR,
#         far=FAR,
#         device=device,
#     )


# if __name__ == "__main__":
#     main()



import torch
import torch.nn as nn

from config import *
from dataset import load_tiny_nerf, load_blender_nerf
from model import TinyNeRF
from train import train


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if DATASET_TYPE == "tiny":
        images, poses, focal, H, W, testimg, testpose = load_tiny_nerf(DATA_PATH, device)
    else:
        images, poses, focal, H, W, testimg, testpose = load_blender_nerf(BLENDER_SCENE_DIR, device, train_views=None)

    model = TinyNeRF().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train(
        model, optimizer,
        images, poses, focal, H, W,
        testimg, testpose,
        SAVE_DIR, N_ITERS, N_SAMPLES,
        N_RAND, PLOT_STEP, NEAR, FAR, device
    )


if __name__ == "__main__":
    main()
