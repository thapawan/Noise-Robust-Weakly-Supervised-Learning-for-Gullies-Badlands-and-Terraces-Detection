# Noise-Robust-Weakly-Supervised-Learning-for-Gullies-Badlands-and-Terraces-Detection
***

```markdown
# Gullies, Terraces & Badlands — Noise‑Robust Weakly Supervised Deep Learning
!License: MIT
!Python 3.10+
!PyTorch
!Status: Research

A reproducible pipeline for **noise‑robust**, **weakly supervised** deep learning to detect **gullies** and **badlands** using heterogeneous community mapping (points & boxes) and multi‑sensor inputs (Sentinel‑2, DEM derivatives).

> **Why this repo?**  
> Expert mapping at global scale is heterogeneous and noisy. This code converts **points** and **boxes** into effective supervision, trains models with **robust losses** (GCE/bootstrapped + focal), and enforces **spatially strict** evaluation (by site & mapper). Outputs include **probability** and **uncertainty** maps suitable for **susceptibility mapping** and **QC triage**.

---

## ✨ Highlights

- **CPU or GPU:** Works on **CPU** out-of-the-box (slower) and **CUDA GPUs** (faster).
- **Weak supervision:** Converts gully points → **soft masks** (Gaussian buffers) and badland/terrace boxes → **MIL regions**.
- **Noise robustness:** Combines **Generalized Cross‑Entropy**, **Focal**, and **Bootstrapped CE**; co‑teaching & hard‑negative mining ready.
- **Spatial rigor:** **Site/mapper stratified** splits to prevent spatial leakage.
- **Uncertainty-aware:** Optional MC‑Dropout/TTA for uncertainty & calibration (ECE).

---

## 🗂️ Repository Structure

```

gully-badlands-weaklysupervised/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ .gitignore
├─ requirements.txt
├─ environment.yml
├─ configs/
│  ├─ default.yaml                 # data, loss, augmentation, metrics
│  ├─ seg\_deeplab\_resnet50.yaml    # model config (swap in DeepLab later)
│  └─ seg\_unet\_efficient.yaml      # alternative config
├─ src/gullies/
│  ├─ **init**.py
│  ├─ utils.py                     # config + seeding
│  ├─ data.py                      # dataset loader (NPY placeholder)
│  ├─ models.py                    # SimpleUNet baseline (plug-in point)
│  ├─ losses.py                    # GCE, Focal, BootstrappedCE
│  ├─ train\_loop.py                # Trainer (CPU/GPU autodetect)
│  └─ eval\_metrics.py              # PR-AUC, etc.
├─ scripts/
│  ├─ prepare\_data.py              # tiling + soft labels (to be filled)
│  ├─ train.py                     # training entrypoint
│  ├─ evaluate.py                  # metric evaluation (expand)
│  └─ infer.py                     # sliding-window inference (expand)
└─ notebooks/
└─ 01\_quickstart.ipynb

````

---

## 🚀 Quick Start

### 1) Clone
```bash
git clone https://github.com/<your-username>/gully-badlands-weaklysupervised.git
cd gully-badlands-weaklysupervised
````

### 2) Create environment

**Conda (recommended):**

```bash
conda env create -f environment.yml
conda activate gully-badlands-weaklysupervised
```

### 3) Install PyTorch (pick one)

**CPU only**

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu
```

**CUDA 11.8**

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1**

```bash
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
```

### 4) Install remaining deps

```bash
pip install -r requirements.txt
```

***

## 📦 Data & Layout

Prepare imagery stacks and labels like this:

    project/
      data/
        tiles/            # preprocessed image tiles (e.g., float32 NPY/GeoTIFF)
        labels/           # soft masks (float), MIL boxes (GeoJSON)
        splits/           # train/val/test JSON with site+mapper stratification
      outputs/
        ckpts/            # checkpoints
        logs/             # training logs
        preds/            # probability/uncertainty GeoTIFFs

*   **Inputs** (customizable in `configs/default.yaml`):
    *   Sentinel‑2 bands & indices: `B02,B03,B04,B08,NDVI,NDMI,BareSoil`
    *   DEM derivatives: `SLOPE,CURV,FLOWACC`
*   **Labels:**
    *   **Gully points** → multi-scale buffers (e.g., 1–3 px) with **Gaussian decay**.
    *   **Badland/terrace boxes** → region-level positives with **MIL** (at least one positive).
    *   **Negatives:** outside any buffers/boxes; add **hard negatives** from false positives later.

> Use `scripts/prepare_data.py` to implement tiling, index computation, DEM derivatives, and label generation (scaffold provided).

***

## ⚙️ Configuration

See `configs/default.yaml` for:

*   `data.bands` and tile size/stride
*   Weak supervision parameters: buffer sizes, Gaussian sigma, MIL minimum positive
*   Loss settings: **GCE** `q`, **Focal** `gamma`, **Bootstrapped CE** `beta`, label smoothing
*   Co‑teaching schedule (toggle on/off)
*   Class/region weights and curriculum

Example model override:

```yaml
# configs/seg_deeplab_resnet50.yaml
model:
  name: deeplabv3plus
  backbone: resnet50
  in_channels: 10
  num_classes: 2
include_dem: true
base_config: configs/default.yaml
```

***

## 🧠 Train

> The trainer uses **GPU if available**, else CPU.

```bash
python scripts/train.py --config configs/seg_deeplab_resnet50.yaml
```

The current baseline uses `SimpleUNet` for quick sanity checks. Swap in DeepLabV3+ or other backbones by extending `src/gullies/models.py` and updating the config.

***

## 📈 Evaluate

Implement metric computation in `scripts/evaluate.py` (scaffold provided):

*   Pixel-level: **PR‑AUC**, **ROC‑AUC**, **mIoU**, **F1\@optimal threshold**
*   Calibration: **ECE**
*   Cross‑region performance: site/mapper holdout

```bash
python scripts/evaluate.py --ckpt outputs/ckpts/best.ckpt --config configs/seg_deeplab_resnet50.yaml
```

***

## 🗺️ Inference

Sliding-window inference over a raster stack (add your I/O logic; scaffold prints TODO):

```bash
python scripts/infer.py --ckpt outputs/ckpts/best.ckpt \
  --raster_stack /path/to/stack.tif \
  --out outputs/preds.tif
```

Recommended outputs:

*   Probability GeoTIFF (float32, \[0,1]).
*   Optional uncertainty (std-dev across MC‑Dropout/TTA).
*   Thresholded mask (for vectorization).

***

## 🧪 Suggested Experiments (for the paper)

1.  **Baseline vs noise‑robust training:** CE vs GCE+Focal+Bootstrapped, same splits.
2.  **Supervision strategy:** points‑only vs points+boxes (MIL) vs boxes‑only.
3.  **Domain generalization:** train in A+B, test in C; augment/style ablations.
4.  **Uncertainty utility:** triage efficiency—% tiles flagged vs % errors captured.
5.  **Input modalities:** S2 only vs S2+DEM vs +ancillary; report PR‑AUC/mIoU deltas.

***

## 🧰 Roadmap

*   [ ] Implement **DeepLabV3+** (torchvision) and config switch
*   [ ] Add **co‑teaching** trainer and **hard negative mining**
*   [ ] Full **prepare\_data.py** (tiling, indices, DEM, soft labels, splits)
*   [ ] **MC‑Dropout/TTA** and **ECE** for uncertainty & calibration
*   [ ] **GeoTIFF I/O** in inference + seamless CRS handling
*   [ ] **GitHub Actions**: CPU smoke test + linting

***

## ❓ FAQ

**Q:** *Can I run this on CPU?*  
**A:** Yes. It’s slower but fully supported. The trainer auto‑detects CUDA.

**Q:** *Do I need per‑pixel labels?*  
**A:** No. This pipeline uses **points** and **boxes** via weak supervision (soft masks + MIL).

**Q:** *How do I avoid spatial leakage?*  
**A:** Build **site/mapper** stratified splits in `data/splits/`. Never mix tiles from the same site between train and val/test.

**Q:** *Which bands should I use?*  
**A:** Start with `B02,B03,B04,B08` + `NDVI,NDMI,BareSoil` + `SLOPE,CURV,FLOWACC`. Adjust in `configs/default.yaml`.

***

## 🤝 Contributing

PRs welcome! Please:

*   Keep functionality modular (config‑driven).
*   Include a short note on **reproducibility** and **spatial split** strategy.
*   Add brief unit tests or a CPU smoke run where feasible.


***

## 📝 License

This project is released under the **MIT License**. See LICENSE.

***

## 🙏 Acknowledgments

We thank the contributors to the global gully/badland mapping initiative and the harmonization team for Phase‑1 data curation. Their efforts make robust, transferable models possible.

***


