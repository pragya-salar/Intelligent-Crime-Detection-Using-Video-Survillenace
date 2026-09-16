# Intelligent Crime Detection and Alert System using Video Surveillance

A skeleton-based violence detection system that combines **ST-GCN (Spatio-Temporal Graph Convolutional Networks)** with a **Transformer Encoder** to detect violent activity from video footage in real time — with a Telegram-based alert pipeline for practical deployment.

---

## Overview

Traditional violence-detection systems rely on raw video pixels — computationally expensive, privacy-invasive, and sensitive to lighting/background changes. This project instead represents human motion as a **skeleton graph** (17 COCO keypoints), making detection background-invariant, privacy-preserving, and lightweight enough for real-time deployment.

**Pipeline:** Video → Pose Estimation → Feature Engineering → Hybrid ST-GCN + Transformer Classifier → Telegram Alert

---

## Key Results

| Dataset | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Hockey Fight | 91.39% | 89.9% | 93.4% | 91.6% |
| Movie Fight | **96.88%** | **100.0%** | 93.7% | 96.8% |

- Zero false positives on the Movie Fight test set (32/32 non-fight clips correctly classified).
- Strong performance on both real-world sports footage (Hockey Fight) and choreographed action sequences (Movie Fight).

---

## How It Works

1. **Pose Estimation** — [YOLO11x-Pose](https://github.com/ultralytics/ultralytics) extracts 17 COCO keypoints per frame from 30 uniformly sampled frames per video clip.
2. **Feature Engineering** — Raw `(x, y, confidence)` keypoints are enriched into a **7-channel** representation per joint:
   - Position `(x, y)`
   - Confidence
   - Velocity `(vx, vy)` — frame-to-frame displacement
   - Bone vectors `(bx, by)` — limb orientation relative to anatomical neighbors
3. **STGCNTransformer Model**
   - 8 ST-GCN blocks (channels: 7 → 64 → 128 → 256) for local spatio-temporal joint modeling
   - 4-layer Transformer Encoder (8 heads, d_model = 256) for global temporal reasoning
   - Classification head → fight / no-fight
4. **Alert System** — On detecting violence, **CrimeAlertBot** sends a Telegram notification with a timestamp, confidence score, and annotated frame.

---

## Tech Stack

- **Deep Learning:** PyTorch, Ultralytics YOLO11x-Pose
- **Data/Compute:** NumPy, OpenCV, Google Colab (Tesla T4 GPU)
- **Evaluation:** Scikit-learn, Matplotlib, Seaborn
- **Deployment:** Telegram Bot API

---

## Dataset

Trained on the combined **Hockey Fight** and **Movie Fight** datasets (Nievas et al., 2011):

| Split | Hockey Fight | Movie Fight | Total |
|---|---|---|---|
| Train | 672 | 168 | 840 |
| Validation | 127 | 33 | 160 |
| Test | 151 | 32 | 183 |

Stratified 70/15/15 split, fixed seed (42), datasets split independently before merging to keep test sets strictly unseen.

> Datasets are not included in this repo due to size/licensing — see `data/README.md` for download instructions and expected folder structure.

---

## Repository Structure

```
├── notebooks/
│   └── crime_detection.ipynb      # end-to-end pipeline (extraction → training → inference)
├── src/                           # (recommended) modularized scripts
│   ├── skeleton_extraction.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── telegram_alert.py
├── models/
│   └── best_model.pth             # trained checkpoint (best validation accuracy, epoch 36)
├── results/
│   ├── training_curves.png
│   └── confusion_matrices.png
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/pragya-salar/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
```

Create a `.env` file (never commit this):
```
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
```

Run skeleton extraction and training via the notebook in `notebooks/`, or the modular scripts in `src/` if refactored.

---

## Model Architecture

```
Input (30, 17, 7)
   → 8x ST-GCN Blocks (7 → 64 → 128 → 256 channels, temporal downsampling at blocks 4 & 7)
   → Global Average Pool (joints)
   → + Positional Encoding
   → 4x Transformer Encoder Layers (8 heads, d_model=256, pre-norm)
   → Global Average Pool (time)
   → Linear(256→128) → GELU → Linear(128→2)
   → Fight / No-Fight
```

Training: AdamW (lr=5e-4, wd=5e-4), linear warmup (10 epochs) + cosine annealing (50 epochs), label smoothing (0.1), 5 data augmentations (horizontal flip, temporal jitter, Gaussian noise, random scale, confidence dropout).

---

## Limitations & Future Work

- Single-person skeleton representation — does not model inter-person spatial relationships (proximity, contact).
- Largest-bounding-box person selection may pick the wrong subject in crowded scenes.
- Future: multi-person graph with inter-person edges, training on larger real-world surveillance datasets for broader generalization.

---

## Contributors

Garima · Pragya Salar · Diya Khatri · Aaryan Yadav
Department of Computer Science and Engineering, Graphic Era (Deemed to be University), Dehradun

---

## License

This project is released under the [MIT License](LICENSE).
