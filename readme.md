# From Gigapixels to Bacteria: *H. pylori* Detection via Active Learning

> **An active learning system for automated *Helicobacter pylori* detection in veterinary whole slide images**
---

## 📄 Abstract

*Helicobacter pylori* is a spiral-shaped bacterium classified as a Class I carcinogen that causes chronic gastritis and gastric cancers in animals. Detecting it requires identifying 2–5 µm bacteria within gigapixel whole slide images (WSIs) — a task that takes approximately 30 minutes per slide and is highly prone to observer fatigue.

Standard supervised approaches fail here because pathologists annotate only enough bacteria to confirm diagnosis, leaving hundreds unmarked. Standard metrics like mAP become unreliable when ground truth is incomplete.

We present an **active learning pipeline** that iteratively refines detection using expert verification and hard negative mining. Over three cycles, our dataset grew from 1,153 to 6,317 samples, and **Impact Precision** — a clinically meaningful metric we propose for incomplete annotation scenarios — improved from **29.6% to 79.3%**. The final deployment model is a **Faster R-CNN** (ResNet-50 + FPN), which at a confidence threshold of 0.75 achieves 79% precision while detecting 287 of 398 verified bacteria.

> **Authors:** Rohan Sanjay Patil, Vidya Padmanabha, Harsha Sathish  
> **Affiliation:** THWS — Technical University of Applied Sciences Würzburg-Schweinfurt, MAI Programme, Germany  
> **Paper:** [HPylori_ICDD2026.pdf](./paper/HPylori_ICDD2026.pdf)

---

## 📁 Repository Structure

```
HPylori_Detection/
│
├── final_models/                        # ← Final trained models & evaluation notebooks
│   ├── Train_RCNN_7_02_it3.ipynb        #   Faster R-CNN final model (evaluation & deployment)
│   ├── train_yolo_it0.ipynb             #   Active Learning — Iteration 0 (baseline)
│   ├── train_yolo_it1.ipynb             #   Active Learning — Iteration 1 (conservative)
│   ├── train_yolo_it2.ipynb             #   Active Learning — Iteration 2 (recall experiment)
│   ├── train_yolo_it3.ipynb             #   Active Learning — Iteration 3 (final YOLO)
│   ├── yolov8s.pt                       #   YOLOv8 small weights (used in AL cycles)
│   ├── yolov8n.pt                       #   YOLOv8 nano weights
│   ├── yolov8m.pt                       #   YOLOv8 medium weights
│   └── yolo11n.pt                       #   YOLOv11 nano weights
│
├── model_tryouts/                       # ← Experiments & training data
│   ├── ActiveLearning_Union_Approach.ipynb
│   ├── Project_Implementation/
│   │   ├── patches/                     #   2,603 × 512×512 image tiles (PNG)
│   │   ├── labels/                      #   2,602 YOLO-format annotation files (TXT)
│   │   ├── YOLO_HPylori_Training.ipynb
│   │   └── RCNN_HPylori_Training.ipynb
│   ├── YOLO_imp/                        #   Earlier YOLO experiments
│   └── RCNN_imp/                        #   Earlier RCNN experiments
│
├── paper/
│   └── HPylori_ICDD2026.pdf             # Conference paper (ICDD 2026)
│
├── poster/                              # Poster assets
│
└── workings/                            # Meeting minutes, proposals, documentation
    ├── minutes/                         # Weekly meeting notes (MOM)
    ├── proposal/                        # Project proposal files
    └── ProgressPresentation/
        └── Documentation/               # Dual model strategy, QuPath integration docs
```

---

## 🔬 Methodology

### The Core Problem

Traditional object detection assumes complete ground truth. In diagnostic histopathology, this assumption breaks down. A pathologist confirming *H. pylori* infection may label 50 bacteria to establish diagnosis — then move to the next slide — leaving 500+ bacteria unlabeled. Training on this sparse data causes two failure modes:

- Models penalised for correctly detecting unlabelled bacteria (treated as false positives)
- Models unable to distinguish bacteria from morphologically similar artifacts (ink, tissue folds, debris)

### Our Solution: Active Learning with Hard Negative Mining

```
┌─────────────────────────────────────────────────────────────┐
│                   Active Learning Loop                       │
│                                                             │
│  1. Train YOLOv8 on annotated patches + provisional neg.    │
│         ↓                                                   │
│  2. Deploy on ~120,000 unannotated WSI patches              │
│         ↓                                                   │
│  3. Expert reviews ~2% flagged candidates                   │
│         ↓                                                   │
│  4. TP  →  New positive training samples                    │
│     FP  →  Hard negatives (most informative failures)       │
│         ↓                                                   │
│  5. Enrich dataset & retrain → repeat                       │
└─────────────────────────────────────────────────────────────┘
```
---

## 📊 Key Results

### Impact Precision — A Clinical Metric for Incomplete Annotations

We introduce **Impact Precision** as the primary evaluation metric:

$$P_{impact} = \frac{\text{Expert-confirmed bacteria}}{\text{Total AI predictions}}$$

**Clinical interpretation:** If $P_{impact}$ = 79%, when the AI flags a region there is a 79% probability a pathologist will confirm *H. pylori*. This directly measures whether AI assistance reduces or increases pathologist workload — the metric standard mAP cannot capture.

### The Metric Paradox

| Iteration | mAP | YOLO Precision | $P_{impact}$ @ 0.15 | $P_{impact}$ @ 0.25 |
|---|---|---|---|---|
| AL 0 (Baseline) | 0.41 | 0.70 | 29.6% | 38% |
| AL 1 (Conservative) | 0.48 | 0.73 | 59.4% | 90% |
| AL 2 (Ratio Failure) | 0.35 | 0.42 | 14.5% | 15% |
| AL 3 (Final YOLO) | 0.33 | 0.40 | 66.8% | 75% |
| **Faster R-CNN (Final)** | — | — | — | **79.3% @ 0.75** |

> The model with the **lowest mAP (AL 3, 0.33) yields the best clinical performance**. Traditional metrics are unreliable under incomplete annotations because they cannot distinguish unlabelled true positives from genuine false positives.

### Dataset Growth

| Iteration | Positives | Hard Negatives | Total | Impact Precision | Key Discovery |
|---|---|---|---|---|---|
| AL 0 | 1,153 | 0 | 1,153 | 29.6% | Ink artifacts |
| AL 1 | 2,150 | 2,372 | 4,522 | 59.4% | Learned ink discrimination |
| AL 2 | 2,412 | 3,708* | 6,120 | 14.5% | Ratio imbalance failure |
| AL 3 | 2,609 | 3,708 | 6,317 | 66.8% | Optimal balance |
| **Faster R-CNN** | 2,609 | 3,708 | 6,317 | **79.3%** | Final deployment |

*Hard negatives retained from AL 1; provisional negatives dropped.

**Total bacteria discovered through active learning: 1,456 previously unmarked instances** across 20 WSIs — suggesting initial expert annotations captured approximately 40% of actual bacteria present.

---

## ⚙️ Model Configurations

### YOLOv8s — Active Learning Cycles

```python
# Architecture
model = "yolov8s"
pretrained_weights = "COCO"

# Training
epochs = 150
batch_size = 16
optimizer = "AdamW"
lr = 0.001          # Decays to 0.0001
weight_decay = 0.0005

# Augmentation (orientation + staining invariance)
rotation     = "±180°"
flip         = "horizontal + vertical"
hsv_hue      = 0.015
hsv_sat      = 0.7
hsv_val      = 0.4
scale        = "0.5× – 1.5×"

# Inference thresholds
conf_active_learning = 0.15   # High recall for candidate generation
conf_deployment      = 0.25   # Balanced precision/recall
nms_iou              = 0.45
```

### Faster R-CNN — Final Deployment Model

```python
# Architecture
backbone    = "ResNet-50 + Feature Pyramid Network (FPN)"
pretrained  = "ImageNet"

# Anchors tuned for 2-5 µm bacteria at 40× magnification
anchor_sizes = [4, 8, 16, 32, 64]   # pixels

# Same augmentations as YOLOv8 for consistency
# Inference
conf_threshold = 0.75    # Clinical deployment
```

### Hardware

Training conducted on **NVIDIA RTX 3090**.  
- YOLOv8 per iteration: ~4–6 hours  
- Faster R-CNN training: ~12–18 hours  
- WSI inference (~120,000 patches): ~15–20 min (batch size 32)

---

## 🚀 Usage

### Dependencies

```bash
pip install torch torchvision ultralytics opencv-python scikit-learn tqdm matplotlib seaborn pillow
```

### Running the Final RCNN Model (Evaluation)

The notebook `final_models/Train_RCNN_7_02_it3.ipynb` contains the full evaluation pipeline, including:

- Loading Faster R-CNN weights (`rcnn_bacteria_best_f1.pth`)
- Running inference on WSI patches
- Generating QuPath-compatible XML annotations
- Threshold sensitivity analysis (precision/recall vs. confidence)
- Expert verification comparison

Update the configuration block at the top of the notebook:

```python
class EvalConfig:
    PROJECT_ROOT = Path("/path/to/your/hpylori_project")
    MODEL_PATH   = PROJECT_ROOT / "rcnn_optimal_final/models/rcnn_bacteria_best_f1.pth"
    PATCHES_DIR  = PROJECT_ROOT / "master-data/separated_patches/test_data_full/images"
```

### Running YOLO Active Learning Iterations

Each `train_yolo_it{n}.ipynb` notebook in `final_models/` corresponds to one active learning cycle. Run them in order (it0 → it1 → it2 → it3). The notebooks handle:

- Dataset loading (patches from `model_tryouts/Project_Implementation/`)
- YOLOv8 training with configurable positive/negative ratio
- Deployment on unlabelled patches
- Generating XML annotations for expert review in QuPath

### Data Format

Patches follow YOLO annotation format:

```
patches/
  593437_svs_x02050_y12811_512.png   ← 512×512 tile, named by WSI ID + grid position
labels/
  593437_svs_x02050_y12811_512.txt   ← YOLO format: class cx cy w h (normalised)
```

Class `0` = *H. pylori* bacterium.

---

## 🗂️ Contents of `final_models/`

| File | Type | Description |
|---|---|---|
| `Train_RCNN_7_02_it3.ipynb` | Notebook | **Final model.** Faster R-CNN evaluation, threshold analysis, and deployment scripts |
| `train_yolo_it0.ipynb` | Notebook | AL Iteration 0 — Baseline YOLOv8 (29.6% precision) |
| `train_yolo_it1.ipynb` | Notebook | AL Iteration 1 — Conservative 1:1 ratio (59.4% precision) |
| `train_yolo_it2.ipynb` | Notebook | AL Iteration 2 — 4:1 ratio experiment (14.5% precision) |
| `train_yolo_it3.ipynb` | Notebook | AL Iteration 3 — Optimal dataset (66.8% precision) |
| `yolov8s.pt` | Weights | YOLOv8 small — used in AL cycles |
| `yolov8n.pt` | Weights | YOLOv8 nano |
| `yolov8m.pt` | Weights | YOLOv8 medium |
| `yolo11n.pt` | Weights | YOLOv11 nano |

> **Note:** The trained Faster R-CNN weights (`rcnn_bacteria_best_f1.pth`) are not included in the repository due to file size. Contact the authors for access.

---

## 📖 Key Concepts

**Whole Slide Image (WSI):** A digitised biopsy slide typically ~100,000 × 100,000 pixels at 40× magnification, requiring patch-based processing.

**Impact Precision ($P_{impact}$):** Fraction of AI-flagged regions confirmed as bacteria by a pathologist. Measures clinical utility under incomplete annotations, where mAP fails.

**Hard Negative Mining:** Rather than discarding false positives, they are added back to training as explicit negative examples — the most informative samples for learning the decision boundary between bacteria and confounding tissue structures.

**Hotspot Extraction:** QuPath-based pipeline that segments glandular tissue and discards irrelevant background/stroma, reducing the patch search space by ~60%.

---

## ⚠️ Limitations

- **Incomplete ground truth:** Despite active learning improvements, annotations remain partial. Some detected "false positives" may be genuine unmarked bacteria.
- **Single stain:** System trained exclusively on H&E-stained slides. Performance on Giemsa, Warthin-Starry, or IHC stains is untested.
- **Veterinary focus:** Developed on companion animal gastric biopsies (LABOKLIN dataset). Applicability to human clinical slides requires validation.
- **LIS integration:** Full clinical deployment requires interfacing with laboratory information systems for automated reporting.
- **Prospective validation:** No prospective clinical trial comparing AI-assisted vs. traditional diagnostic workflows has been conducted.

---

## 🔮 Future Work

- Extend to multi-stain protocols (Giemsa, Warthin-Starry, immunohistochemical staining)
- Integrate with Laboratory Information Systems (LIS/LIMS) for automated reporting
- Prospective clinical validation comparing AI-assisted vs. standard diagnostic workflows
- Continue active learning iterations on additional WSIs to further improve precision and recall
- Uncertainty quantification to identify slides requiring additional expert review

---

## 🙏 Acknowledgements

- **[LABOKLIN](https://www.laboklin.de)** — for providing the veterinary gastric biopsy dataset and expert pathological annotations
- **Philipp Ockermann** — critical support throughout the active learning process
- **Prof. Dr. Magda Gregorová** — project supervision and guidance
- **THWS (Technical University of Applied Sciences Würzburg-Schweinfurt)** — computational resources and research infrastructure

---

## 📚 Citation

If you use this code or methodology, please cite:

```bibtex
@inproceedings{sathish2026hpylori,
  title     = {From Gigapixels to Bacteria: An Active Learning System for
               \textit{Helicobacter pylori} Detection in Whole Slide Images},
  author    = {Patil, Rohan Sanjay and Padmanabha, Vidya and Sathish, Harsha},
  booktitle = {International Conference on Applied Informatics Imagination, Creativity, Design, Development - ICDD},
  year      = {2026},
  address   = {W{\"u}rzburg, Germany},
  institution = {THWS, MAI}
}
```

---

## 📬 Contact

For questions, collaboration inquiries, or access to trained model weights:

| Name | Role |
|---|---|
| Rohan Sanjay Patil | Author, THWS MAI | rohansanjay.patil@thws.de
| Vidya Padmanabha | Author, THWS MAI |vidya.padmanabha@thws.de
| Harsha Sathish | Author, THWS MAI | harsha.sathish@thws.de

---

*Developed as part of the MAI Project Module at THWS — February 2026*
