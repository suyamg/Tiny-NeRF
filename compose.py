import cv2
import numpy as np

# 너가 생성한 NeRF 이미지
nerf_img = cv2.imread("/home/junho/suyang/Nerf/Tiny_NeRF/TEST/TEST_1/train2_(30)/iter_20000_view_1.png")

# 방금 제공한 Unity-style 배경 이미지
bg_img = cv2.imread("/home/junho/suyang/Nerf/Tiny_NeRF/datasets/image.png")

# 크기를 동일하게 맞추기
h, w = nerf_img.shape[:2]
bg_img = cv2.resize(bg_img, (w, h))

# Alpha blending (0.0~1.0 조절 가능)
alpha = 0.65
blended = cv2.addWeighted(nerf_img, alpha, bg_img, 1 - alpha, 0)

cv2.imwrite("composited_result.png", blended)
print("Saved → composited_result.png")
