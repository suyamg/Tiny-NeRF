# Tiny-NeRF (PyTorch)

> A minimal PyTorch implementation of NeRF, based on the Tiny NeRF tutorial.


![TinyNeRF Result]([result/nerf_training.gif](https://github.com/suyamg/Tiny-NeRF/blob/main/result/final_result(1)/nerf_training.gif))

---

## 🧠 프로젝트 소개 (Overview)

**Neural Radiance Fields (NeRF)** 를 아주 작게 축소한 버전으로,  
다음 개념들을 코드로 직접 따라가볼 수 있게 만든 구현입니다.

- 카메라 포즈 → 픽셀별 Ray 생성
- Positional Encoding (Fourier Features)
- MLP를 이용한 색/밀도 예측
- Volume Rendering으로 Ray 색 합성
- MSE loss + Adam으로 NeRF 학습

---

## 📂 Project Structure

```
Tiny-NeRF/
  ├── config.py      # 경로, 하이퍼파라미터 설정
  ├── dataset.py     # tiny_nerf_data.npz 로딩
  ├── model.py       # Positional Encoding + TinyNeRF MLP
  ├── rays.py        # get_rays, cumprod_exclusive (Ray 관련 유틸)
  ├── render.py      # render_rays: NeRF 볼륨 렌더링
  ├── train.py       # 학습 루프 + 평가 + 이미지/GIF 저장
  ├── main.py        # 엔트리 포인트 (python main.py)
  ├── Data/          # tiny_nerf_data.npz (데이터 파일 위치)
  └── result/        # 학습 중 저장되는 렌더링 결과, GIF
```

```
git clone https://github.com/suyamg/Tiny-NeRF.git
cd Tiny-NeRF

pip install torch torchvision torchaudio
pip install numpy imageio
```


Python 3.8+ 기준, 아래 패키지가 필요합니다:
- torch
- torchvision 
- numpy
- imageio

```
cd Tiny-NeRF
python main.py
```



## References
- Original NeRF paper: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (Mildenhall et al., ECCV 2020)
- Tiny NeRF tutorial (Colab)
