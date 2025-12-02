# import os
# import glob
# import imageio
# import torch
# import torch.nn.functional as F

# from rays import get_rays
# from render import render_rays


# def mse2psnr(mse: torch.Tensor, device) -> float:
#     return (-10.0 * torch.log(mse) / torch.log(torch.tensor([10.0], device=device))).item()


# def make_gif(save_dir: str,
#              pattern: str = "nerf_result_iter_*.png",
#              output_name: str = "nerf_training.gif"):
#     files = sorted(
#         glob.glob(os.path.join(save_dir, pattern)),
#         key=os.path.getmtime
#     )
#     if not files:
#         print("⚠️ GIF 생성 실패: 저장된 이미지가 없습니다.")
#         return
#     images = [imageio.imread(f) for f in files]
#     imageio.mimsave(os.path.join(save_dir, output_name), images, fps=2)
#     print(f"GIF 저장: {os.path.join(save_dir, output_name)}")


# def train(model,
#           optimizer,
#           images,
#           poses,
#           focal,
#           H,
#           W,
#           testimg,
#           testpose,
#           save_dir: str,
#           n_iters: int,
#           n_samples: int,
#           N_rand: int,
#           plot_step: int,
#           near: float,
#           far: float,
#           device):

#     psnrs = []

#     for i in range(n_iters):
#         # 1) 랜덤 train 이미지 선택 (마지막 index 99는 test용이라 제외)
#         img_i = torch.randint(low=0, high=images.shape[0] - 1, size=(1,)).item()
#         target_full = images[img_i]   # [H,W,3]
#         pose        = poses[img_i]

#         # 2) 전체 레이 생성
#         rays_o_full, rays_d_full = get_rays(H, W, focal, pose, device=device)

#         # 3) flatten 후 N_rand개 샘플링
#         rays_o_flat = rays_o_full.reshape(-1, 3)
#         rays_d_flat = rays_d_full.reshape(-1, 3)
#         target_flat = target_full.reshape(-1, 3)

#         select_inds = torch.randperm(H * W, device=device)[:N_rand]
#         rays_o = rays_o_flat[select_inds]
#         rays_d = rays_d_flat[select_inds]
#         target = target_flat[select_inds]

#         # 4) forward + loss + backward
#         model.train()
#         rgb, depth, acc = render_rays(
#             model, rays_o, rays_d,
#             near=near, far=far,
#             n_samples=n_samples,
#             rand=True,
#             device=device
#         )
#         loss = F.mse_loss(rgb, target)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         # 5) 정기 평가 & 이미지 저장
#         if i % plot_step == 0:
#             model.eval()
#             with torch.no_grad():
#                 rays_o_eval, rays_d_eval = get_rays(H, W, focal, testpose, device=device)
#                 rays_o_eval = rays_o_eval.reshape(-1, 3)
#                 rays_d_eval = rays_d_eval.reshape(-1, 3)

#                 rgb_eval_flat, depth_eval_flat, acc_eval_flat = render_rays(
#                     model, rays_o_eval, rays_d_eval,
#                     near=near, far=far,
#                     n_samples=n_samples,
#                     rand=False,
#                     device=device
#                 )
#                 rgb_eval = rgb_eval_flat.reshape(H, W, 3)
#                 test_loss = F.mse_loss(rgb_eval, testimg)
#                 psnr = mse2psnr(test_loss, device)
#                 psnrs.append(psnr)

#                 print(f"[Iter {i}] train_loss={loss.item():.6f}, test_mse={test_loss.item():.6f}, PSNR={psnr:.2f}")

#                 img = (rgb_eval.clamp(0.0, 1.0).cpu().numpy() * 255).astype("uint8")
#                 imageio.imwrite(os.path.join(save_dir, f"nerf_result_iter_{i}.png"), img)

#     make_gif(save_dir)



import os
import imageio
import torch
import torch.nn.functional as F

from render import render
from utils import augment_image, jitter_rays, get_rays
from utils import compute_psnr


def train(model, optimizer, images, poses, focal,
          H, W, testimg, testpose,
          save_dir, n_iters, n_samples, N_rand,
          plot_step, near, far, device):

    os.makedirs(save_dir, exist_ok=True)

    for i in range(n_iters):

        # -------------------------
        # 1) Random training view
        # -------------------------
        idx = torch.randint(images.shape[0], (1,))
        target_full = images[idx][0]
        pose = poses[idx][0]

        # -------------------------
        # 2) Rays from this image
        # -------------------------
        rays_o_full, rays_d_full = get_rays(H, W, focal, pose)

        rays_o_full = rays_o_full.reshape(-1, 3)
        rays_d_full = rays_d_full.reshape(-1, 3)
        target_full = target_full.reshape(-1, 3)

        sel = torch.randint(H * W, (N_rand,))
        rays_o = rays_o_full[sel]
        rays_d = rays_d_full[sel]
        target = target_full[sel]

        # -------------------------
        # 3) Data augmentation
        # -------------------------
        target = augment_image(target)
        rays_o, rays_d = jitter_rays(rays_o, rays_d)

        # -------------------------
        # 4) Training step
        # -------------------------
        optimizer.zero_grad()
        rgb, depth, acc = render(model, rays_o, rays_d, near, far, n_samples, device)
        loss = F.mse_loss(rgb, target)
        loss.backward()
        optimizer.step()

        # -------------------------
        # 5) Evaluation & Saving
        # -------------------------
        if i % plot_step == 0:
            print(f"[Iter {i}] loss={loss.item():.5f}")

            # ---- test view ----
            model.eval()
            with torch.no_grad():

                eval_rays_o, eval_rays_d = get_rays(H, W, focal, testpose)
                eval_rays_o = eval_rays_o.reshape(-1, 3)
                eval_rays_d = eval_rays_d.reshape(-1, 3)

                rgb_eval, depth_eval, acc_eval = render(model, eval_rays_o, eval_rays_d,
                           near, far, n_samples, device)

                rgb_eval = rgb_eval.reshape(H, W, 3)

                psnr = compute_psnr(rgb_eval, testimg)
                print(f" → Test PSNR: {psnr:.3f}")

                # Save image
                img = (rgb_eval.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                imageio.imwrite(os.path.join(save_dir,
                    f"iter_{i:05d}.png"), img)

            model.train()

    print("Training done.")

