# 🌓 Apollo 17 3D Scene Reconstruction with Gaussian Splatting & Photogrammetry

**Assignment 4** focuses on reconstructing 3D lunar surface scenes from Apollo 17 image data. We apply a two-stage process: first, traditional photogrammetry using COLMAP, followed by advanced novel view synthesis via **Gaussian Splatting**. The goal is to assess scene fidelity and augmentation effectiveness using both **visual inspection** and **quantitative metrics**.

## 🔧 Pipeline Overview

### Step 1: Preparing Apollo Image Dataset

- Obtained 15 high-resolution Apollo 17 `.png` images.
- Standardized filenames and resolutions for COLMAP compatibility.

### Step 2–3: COLMAP Sparse Reconstruction

- Extracted keypoints and matched features.
- Generated sparse 3D structure using COLMAP.
  
### Step 4: Exporting COLMAP Results for Splatting

- Exported:
  - `images/`, `cameras.txt`, `images.txt`, `points3D.txt`
- Placed inside `data/apollo17/` for downstream training.

### Step 5: Dense & Textured Modeling

- Generated a textured mesh and dense point cloud for comparison.

---

## 🌌 Gaussian Splatting: Training Phase

### Environment Setup

```bash
conda create -n splatter310 python=3.10
conda activate splatter310
pip install -r requirements.txt
```

#### Known Issues

- ❗ *`plyfile` version conflict*: Resolved via:
  ```bash
  pip install plyfile==0.8.1
  ```
- ❗ *PyTorch crash with NumPy >= 2.0*: Fixed by downgrading:
  ```bash
  pip install numpy==1.24.4
  ```

### Training Command

```bash
python train.py -s data/apollo17
```

- Trained up to **30,000 iterations**.
- Output point cloud saved to:
  - `output/000fab2b-8/point_cloud/iteration_30000/point_cloud.ply`

---

## 🖼️ Re-rendering Original Views

### Run Rendering Script

```bash
python render.py -m output/000fab2b-8 -s data/apollo17 --skip_train --iteration 30000
```

- Used COLMAP pose data to reconstruct original 15 views.
- Results:
  - **Generated Views**: `renders/`
  - **Ground Truth**: `gt/`
- Observations: Rendered and GT images were visually very similar—minor variations are perceptible upon rapid switching.

#### Ground Truth Example  
![AS17-137-20905HR](https://github.com/user-attachments/assets/44191935-58e4-48b7-a455-411be76c19e0)

#### Reconstructed View  
![00002](https://github.com/user-attachments/assets/15b6c500-fd4e-462e-94a0-3fb413ad5574)

---

## 📊 View Comparison Metrics

To evaluate quality, we used **PSNR** and **SSIM** scores via `scikit-image`.

```bash
python Assignment4.py
```
---

## ✨ Generating Novel Views (Interpolation)

### Steps:

1. Parsed `images.txt` to extract camera centers.
2. Ran PCA to compute 3 major axes of variation.
3. Synthesized 10 new camera poses by interpolating around the centroid.
4. Stored them in:
   - `novel_poses.json`

### Rendering with Nerfstudio-Compatible Script

```bash
python nerfstudio/nerfstudio/scripts/gaussian_splatting/render.py camera-path   --model-path output/000fab2b-8   --camera-path-filename novel_poses.json   --output-path output/000fab2b-8/novel_renders/pca_poses/   --output-format images
```

#### ❗ Issue: All Rendered Views Were Black

Even after reusing the original `images.txt` poses, the generated images appeared fully black—likely due to incorrect rendering bounds or point cloud size mismatch.

---

## 🆚 COLMAP vs. Meshroom + Augmented Views

- Constructed mesh models using:
  - COLMAP baseline (15 Apollo views)
  - Meshroom with additional 10 novel views

---

## 📈 Evaluation Summary

### Quantitative

- PSNR / SSIM scores:
  - GT vs Reconstructed
  - Baseline vs Augmented Meshes

### Qualitative

- 3D mesh visual comparison
- Coverage visibly improves with novel poses

---

## ⚙️ Requirements

```bash
python==3.10
torch==2.0+
numpy==1.24.4
plyfile==0.8.1
scikit-image
opencv-python
Pillow
```

---

## 📚 Credits & References

This project adapts and builds upon the following sources:

- 📌 [Inria 3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- 📌 [COLMAP Structure-from-Motion](https://colmap.github.io/)
- 📌 [NASA Apollo Image Archive](https://www.hq.nasa.gov/alsj/a17/images17.html)
