import os

DATA_PATH = "/home/junho/suyang/Nerf/Tiny_NeRF/Data/tiny_nerf_data.npz"
# SAVE_DIR  = "/home/junho/suyang/Nerf/result"

DATASET_TYPE = "blender"      # "tiny" 또는 "blender"
BLENDER_SCENE_DIR = "/home/junho/suyang/Nerf/Tiny_NeRF/datasets/nerf_synthetic/mic"
SAVE_DIR  = "/home/junho/suyang/Nerf/Tiny_NeRF/result/UROP/result_1/ship_1"

os.makedirs(SAVE_DIR, exist_ok=True)

N_ITERS    = 10001        # 학습 step 수
N_SAMPLES  = 64           # 레이당 샘플 수
N_RAND     = 1024         # 한 step에 학습할 Ray 개수
PLOT_STEP  = 500         
LR         = 5e-4         
NEAR       = 2.0        
FAR        = 6.0          
SEED       = 42       
TRAIN_VIEW_COUNT = 50     