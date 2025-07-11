# Point-Denoiser
LiDAR Point cloud denoising

# 🌀 Point-MAE Denoising Model

This repository contains code for a point cloud denoising model based on **Point-MAE** with a ViT backbone, designed to learn robust representations from noisy 3D point clouds.

---

## 📦 Requirements

Make sure your environment meets the following requirements:

- Python 3.10
- CUDA-compatible GPU
- PyTorch (version compatible with your CUDA)
- Other Python dependencies listed in `requirements.txt`

Install the required packages with:

```bash
pip install -r requirements.txt
```
### Pre-train the ViT Encoder-Decoder on ShapeNet55 Dataset:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/pretrain_Point_MAE_ViT.yaml --exp_name pretrain_ViT_Transformer
```

### Train the Classification Head:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --start_ckpts experiments/pretrain_Point_MAE_ViT/cfgs/pretrain_ViT_Transformer/ckpt-last.pth
--config cfgs/denoiseClassificationSpherePreTrainingFull.yaml --exp_name denoiseClassification
```
![Results](Denoising_pic.jpg)

### Test the trained Denoiser on ModelNet40:
```bash
python main_vis.py --test \
  --ckpts experiments/denoiseClassificationSpherePreTrainingFull/cfgs/denoiseClassification/ckpt-last.pth \
  --config cfgs/denoiseClassificationSpherePreTrainingFullWithModelNet40.yaml \
  --exp_name denoiseClassificationTest
```
![Results](Denoising_pic_2.jpg)



