# import numpy as np
# import torch


# def load_tiny_nerf(path: str, device):
#     """
#     tiny_nerf_data.npz 로딩.

#     return:
#       images    : [N, H, W, 3]  (torch.float32)
#       poses     : [N, 4, 4]
#       focal     : float
#       H, W      : int
#       testimg   : [H, W, 3]
#       testpose  : [4, 4]
#     """
#     data = np.load(path)
#     images = torch.tensor(data["images"], dtype=torch.float32, device=device)
#     poses  = torch.tensor(data["poses"],  dtype=torch.float32, device=device)
#     focal  = float(data["focal"])

#     H, W = images.shape[1:3]
#     H, W = int(H), int(W)

#     testimg, testpose = images[99], poses[99]
#     return images, poses, focal, H, W, testimg, testpose
import os
import json
import numpy as np
import torch
from PIL import Image


# ===========================
# Tiny NeRF Loader
# ===========================
def load_tiny_nerf(path, device="cpu"):
    data = np.load(path)
    images = torch.tensor(data["images"], dtype=torch.float32).to(device)
    poses = torch.tensor(data["poses"], dtype=torch.float32).to(device)
    focal = float(data["focal"])

    H, W = images.shape[1], images.shape[2]
    testimg = images[0]
    testpose = poses[0]

    return images, poses, focal, H, W, testimg, testpose


# ===========================
# Blender Synthetic Dataset Loader
# ===========================
def load_blender_nerf(scene_dir, device="cpu", train_views=None):
    def load_split(split):
        json_path = os.path.join(scene_dir, f"transforms_{split}.json")
        with open(json_path, "r") as f:
            meta = json.load(f)

        imgs, poses = [], []
        for frame in meta["frames"]:
            file_path = os.path.join(scene_dir, frame["file_path"] + ".png")
            img = Image.open(file_path)
            img = np.array(img).astype(np.float32) / 255.0
            if img.shape[-1] == 4:
              img = img[..., :3]

            imgs.append(img)
            poses.append(frame["transform_matrix"])

        return np.array(imgs), np.array(poses), meta

    train_imgs, train_poses, meta = load_split("train")

    if train_views is not None:
        train_imgs = train_imgs[:train_views]
        train_poses = train_poses[:train_views]

    H, W = train_imgs[0].shape[:2]

    if "camera_angle_x" in meta:
        focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])
    else:
        focal = meta["focal"]

    test_imgs, test_poses, _ = load_split("test")

    images = torch.tensor(train_imgs, dtype=torch.float32).to(device)
    poses = torch.tensor(train_poses, dtype=torch.float32).to(device)

    testimg = torch.tensor(test_imgs[0], dtype=torch.float32).to(device)
    testpose = torch.tensor(test_poses[0], dtype=torch.float32).to(device)

    return images, poses, focal, H, W, testimg, testpose
