import os, json, glob, csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
from torchvision import transforms as T
from torchmetrics import StructuralSimilarityIndexMeasure
import lpips

# =========================================================
# CONFIG
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_DIR = "/home/junho/suyang/Nerf/Tiny_NeRF/datasets/nerf_synthetic/materials"
SAVE_DIR = "/home/junho/suyang/Nerf/Tiny_NeRF/TEST/TEST_3"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 실험 옵션 ---
N_train = 50              # 실험 1: 학습 이미지 수 조절 (2, 8, 16, 30)
USE_AUG = False            # 실험 2: augmentation 적용 여부
N_EVAL_VIEWS = 10         # 평가 view 수
LOG_INTERVAL = 1000        # 저장 주기

# =========================================================
# Metric 준비
# =========================================================
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
lpips_metric = lpips.LPIPS(net='alex').to(device)  # perceptual loss


# =========================================================
# DATA LOADING
# =========================================================
def load_nerf_synthetic(path, n_train=None):
    with open(os.path.join(path, "transforms_train.json")) as f:
        meta = json.load(f)

    imgs = []
    poses = []

    for frame in meta["frames"]:
        fname = os.path.join(path, frame["file_path"] + ".png")

        img = imageio.imread(fname) / 255.0
        if img.shape[-1] == 4:      # RGBA → RGB
            img = img[..., :3]

        imgs.append(img)
        poses.append(np.array(frame["transform_matrix"]))

    imgs = np.array(imgs, dtype=np.float32)
    poses = np.array(poses, dtype=np.float32)

    if n_train:
        imgs = imgs[:n_train]
        poses = poses[:n_train]

    H, W = imgs.shape[1], imgs.shape[2]
    focal = 0.5 * W / np.tan(0.5 * meta["camera_angle_x"])

    print(f"Loaded {len(imgs)} training images")
    return imgs, poses, H, W, focal



images_np, poses_np, H, W, focal = load_nerf_synthetic(DATASET_DIR, N_train)
images = torch.Tensor(images_np).to(device)
poses = torch.Tensor(poses_np).to(device)

# 평가용 view 선택
eval_ids = np.linspace(0, len(images)-1, N_EVAL_VIEWS).astype(int)
eval_images = images[eval_ids]
eval_poses = poses[eval_ids]


# =========================================================
# RAYS
# =========================================================
def get_rays(H, W, focal, pose):
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                          torch.arange(H, dtype=torch.float32))
    i, j = i.t().to(device), j.t().to(device)

    dirs = torch.stack([(i - W * 0.5) / focal,
                        -(j - H * 0.5) / focal,
                        -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * pose[:3, :3], -1)
    rays_o = pose[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


# =========================================================
# ENCODING
# =========================================================
def positional_encoder(x, L=6):
    out = [x]
    for i in range(L):
        out.append(torch.sin(2**i * x))
        out.append(torch.cos(2**i * x))
    return torch.cat(out, -1)


# =========================================================
# RENDER
# =========================================================
def cumprod_exclusive(t):
    cp = torch.cumprod(t, -1)
    cp = torch.roll(cp, 1, -1)
    cp[..., 0] = 1.
    return cp

def render(model, rays_o, rays_d, near, far, n_samples, rand=False):
    def batchify(fn, chunk=1024*32):
        return lambda x: torch.cat([fn(x[i:i+chunk]) for i in range(0, x.shape[0], chunk)], 0)

    z = torch.linspace(near, far, n_samples).to(device)

    if rand:
        mids = 0.5 * (z[..., 1:] + z[..., :-1])
        lower = torch.cat([z[..., :1], mids], -1)
        upper = torch.cat([mids, z[..., -1:]], -1)
        t_rand = torch.rand(z.shape).to(device)
        z = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z[..., :, None]
    flat = positional_encoder(pts.reshape(-1, 3))

    raw = batchify(model)(flat)
    raw = raw.reshape(list(pts.shape[:-1]) + [4])

    sigma = F.relu(raw[..., 3])
    rgb = torch.sigmoid(raw[..., :3])

    dists = torch.cat([z[..., 1:] - z[..., :-1],
                       torch.tensor([1e10], device=device).expand(z[..., :1].shape)], -1)
    alpha = 1 - torch.exp(-sigma * dists)
    weights = alpha * cumprod_exclusive(1 - alpha + 1e-10)

    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    return rgb_map


# =========================================================
# MODEL
# =========================================================
class TinyNeRF(nn.Module):
    def __init__(self, h=128, L=6):
        super().__init__()
        self.l1 = nn.Linear(3+3*2*L, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 4)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)


# =========================================================
# DATA AUGMENTATION
# =========================================================
augment = T.Compose([
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.15, contrast=0.15)
])

def apply_aug(img):
    img = img.permute(2,0,1)
    img = augment(img)
    return img.permute(1,2,0)


# =========================================================
# TRAIN
# =========================================================
def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.tensor([10.], device=device))
  
def evaluate(model, n_samples=32, eval_downscale=4, chunk=8192):
    psnrs, ssims, lpipss = [], [], []

    H_low = H // eval_downscale
    W_low = W // eval_downscale
    focal_low = focal / eval_downscale

    for gt_full, pose in zip(eval_images, eval_poses):
        
        # downscale GT (for fair comparison)
        gt = F.interpolate(
            gt_full.permute(2,0,1)[None],
            size=(H_low, W_low),
            mode='area'
        )[0].permute(1,2,0)

        # generate low-res rays
        rays_o, rays_d = get_rays(H_low, W_low, focal_low, pose)

        # flatten rays
        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)

        # -------- chunk evaluation --------
        preds = []
        for i in range(0, len(rays_o), chunk):
            ro = rays_o[i:i+chunk]
            rd = rays_d[i:i+chunk]
            rgb = render(model, ro, rd, 2., 6., n_samples)
            preds.append(rgb)
        pred = torch.cat(preds, 0).reshape(H_low, W_low, 3)

        # -------- metrics --------
        loss = F.mse_loss(pred, gt)
        psnrs.append(mse2psnr(loss).item())

        ssims.append(
            ssim_metric(pred.permute(2,0,1)[None], gt.permute(2,0,1)[None]).item()
        )

        # LPIPS uses fixed 256 size
        pred_lp = F.interpolate(pred.permute(2,0,1)[None], size=256)
        gt_lp = F.interpolate(gt.permute(2,0,1)[None], size=256)
        lpipss.append(lpips_metric(pred_lp, gt_lp).item())

    return np.mean(psnrs), np.mean(ssims), np.mean(lpipss)


def train(model, optimizer, iters=20001):
    N_rand = 512 #1024
    n_samples = 32   # OOM 방지 위해 64 → 32

    # -------------------------------------
    # CSV LOG WRITERS (open once)
    # -------------------------------------
    psnr_log = open(os.path.join(SAVE_DIR, "psnr_log.csv"), "w", newline="")
    ssim_log = open(os.path.join(SAVE_DIR, "ssim_log.csv"), "w", newline="")
    lpips_log = open(os.path.join(SAVE_DIR, "lpips_log.csv"), "w", newline="")
    aug_log = open(os.path.join(SAVE_DIR, "augmentation_record.csv"), "w", newline="")

    psnr_writer  = csv.writer(psnr_log)
    ssim_writer  = csv.writer(ssim_log)
    lpips_writer = csv.writer(lpips_log)
    aug_writer   = csv.writer(aug_log)

    psnr_writer.writerow(["iter", "psnr"])
    ssim_writer.writerow(["iter", "ssim"])
    lpips_writer.writerow(["iter", "lpips"])
    aug_writer.writerow(["iter", "use_aug", "train_view_id"])

    # -------------------------------------
    # Train Loop
    # -------------------------------------
    for it in range(iters):

        # random train view
        idx = np.random.randint(len(images))
        target = images[idx]
        pose = poses[idx]

        # augmentation
        if USE_AUG:
            target = apply_aug(target)

        # log augmentation usage
        aug_writer.writerow([it, USE_AUG, int(idx)])

        # create rays
        rays_o, rays_d = get_rays(H, W, focal, pose)
        rays_o = rays_o.reshape(-1,3)
        rays_d = rays_d.reshape(-1,3)
        target_flat = target.reshape(-1,3)

        # random sample N rays
        select = np.random.choice(H*W, N_rand, replace=False)

        rgb = render(
            model,
            rays_o[select], rays_d[select],
            2., 6., n_samples,
            rand=True
        )

        loss = F.mse_loss(rgb, target_flat[select])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # -------------------------------------
        # Logging + Evaluation
        # -------------------------------------
        if it % LOG_INTERVAL == 0:

            psnr, ssim, lpips_v = evaluate(model)

            psnr_writer.writerow([it, psnr])
            ssim_writer.writerow([it, ssim])
            lpips_writer.writerow([it, lpips_v])

            print(f"[{it}] Loss: {loss:.5f} | PSNR: {psnr:.2f} | SSIM: {ssim:.3f} | LPIPS: {lpips_v:.3f}")

            # -------------------------------------
            # SAFE IMAGE SAVING (OOM-free)
            # -------------------------------------
            EVAL_DOWNSCALE = 4
            H_low = H // EVAL_DOWNSCALE
            W_low = W // EVAL_DOWNSCALE
            focal_low = focal / EVAL_DOWNSCALE
            chunk = 8192

            for vid, (gt_full, e_pose) in enumerate(zip(eval_images, eval_poses)):

                # low-res rays
                rays_o_eval, rays_d_eval = get_rays(H_low, W_low, focal_low, e_pose)
                rays_o_eval = rays_o_eval.reshape(-1,3)
                rays_d_eval = rays_d_eval.reshape(-1,3)

                # chunk rendering
                preds = []
                for i in range(0, len(rays_o_eval), chunk):
                    ro = rays_o_eval[i:i+chunk]
                    rd = rays_d_eval[i:i+chunk]
                    rgb = render(model, ro, rd, 2., 6., n_samples)
                    preds.append(rgb)

                pred = torch.cat(preds, 0).reshape(H_low, W_low, 3)

                img = (pred.detach().cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(os.path.join(SAVE_DIR, f"iter_{it}_view_{vid}.png"), img)

    # -------------------------------------
    # END Training → Close Files
    # -------------------------------------
    psnr_log.close()
    ssim_log.close()
    lpips_log.close()
    aug_log.close()

    # Save final model
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "tinynerf_final.pt"))

def make_gif(output_name="nerf_training.gif"):
    # 저장된 이미지 파일들 (view_0만 사용)
    files = sorted(
        glob.glob(os.path.join(SAVE_DIR, "iter_*_view_0.png")),
        key=os.path.getmtime
    )

    if not files:
        print("❗ GIF 생성 실패: 저장된 렌더링 이미지가 없습니다.")
        return

    frames = [imageio.imread(f) for f in files]
    gif_path = os.path.join(SAVE_DIR, output_name)
    imageio.mimsave(gif_path, frames, fps=2)

    print(f"🎉 GIF 저장 완료 → {gif_path}")


# =========================================================
# EXECUTE
# =========================================================
model = TinyNeRF().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
train(model, optimizer)
make_gif("nerf_training.gif")