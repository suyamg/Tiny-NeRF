import numpy as np
import torch


def load_tiny_nerf(path: str, device):
    """
    tiny_nerf_data.npz 로딩.

    return:
      images    : [N, H, W, 3]  (torch.float32)
      poses     : [N, 4, 4]
      focal     : float
      H, W      : int
      testimg   : [H, W, 3]
      testpose  : [4, 4]
    """
    data = np.load(path)
    images = torch.tensor(data["images"], dtype=torch.float32, device=device)
    poses  = torch.tensor(data["poses"],  dtype=torch.float32, device=device)
    focal  = float(data["focal"])

    H, W = images.shape[1:3]
    H, W = int(H), int(W)

    testimg, testpose = images[99], poses[99]
    return images, poses, focal, H, W, testimg, testpose
