# Final Models - H. pylori Detection System

This folder contains the final trained models and training notebooks for the *Helicobacter pylori* detection system developed through an active learning pipeline. The system is designed to detect 2-5 μm bacteria within gigapixel whole slide images (WSIs) of veterinary gastric biopsies stained with Hematoxylin and Eosin (H&E).

## 📋 Overview

The detection system was developed through an iterative active learning approach that progressively refined the training dataset with expert verification. This resulted in:

- **Dataset Growth**: From 1,153 initial annotations to 6,317 samples (2,609 positives + 3,708 hard negatives)
- **Performance Improvement**: Impact Precision increased from 29.6% (baseline) to 79.3% (final model)
- **Discovery**: 1,456 previously unmarked bacteria identified across 20 WSIs

For detailed methodology and results, refer to the paper in `paper/Final_submission.pdf`.

---

## 📁 Files in This Directory

### Training Notebooks

#### YOLOv8 Active Learning Iterations

1. **`train_yolo_it0.ipynb`** - Active Learning Iteration 0 (Baseline)
   - Initial model trained on 1,153 expert-annotated positive samples
   - Used random tissue patches as provisional negatives
   - **Result**: 29.6% Impact Precision
   - **Key Discovery**: 2,372 ink artifacts misclassified as bacteria
   - **Dataset Output**: 2,150 positives + 2,372 hard negatives

2. **`train_yolo_it1.ipynb`** - Active Learning Iteration 1 (Conservative)
   - Trained with 1:1 positive-to-negative ratio
   - **Dataset**: 2,150 positives + 2,372 hard negatives
   - **Result**: 59.4% Impact Precision (100% precision @ confidence > 0.5)
   - **Behavior**: Model became conservative, learned to discriminate bacteria from ink artifacts
   - **Trade-off**: Higher precision but lower recall

3. **`train_yolo_it2.ipynb`** - Active Learning Iteration 2 (Recall Recovery Experiment)
   - Attempted to recover recall with 4:1 positive-to-negative ratio
   - **Dataset**: 2,412 positives + 603 sampled negatives
   - **Result**: 14.5% Impact Precision (experiment failed)
   - **Lesson**: Insufficient hard negatives caused model to revert to aggressive detection
   - **Value**: Discovered 1,157 new diverse artifact types (tissue folds, cellular debris, staining irregularities)

4. **`train_yolo_it3.ipynb`** - Active Learning Iteration 3 (Final YOLOv8)
   - Final YOLOv8 model with optimized positive-to-negative ratio
   - **Dataset**: 2,609 positives + 3,708 hard negatives (6,317 total samples)
   - **Result**: 66.8% Impact Precision (79.9% @ confidence > 0.25)
   - **Deployment**: Recommended threshold of 0.25 for comprehensive coverage

#### Faster R-CNN Final Model

5. **`Train_RCNN_7_02_it3.ipynb`** - Final Deployment Model
   - Two-stage detector (Faster R-CNN with ResNet-50 + FPN backbone)
   - Trained on the enriched dataset from Active Learning Iteration 3
   - **Architecture**: ResNet-50 backbone with Feature Pyramid Network (FPN)
   - **Anchors**: [4, 8, 16, 32, 64] pixels (tuned for bacterial dimensions at 40× magnification)
   - **Result**: 79% Impact Precision @ confidence threshold 0.75
   - **Advantage**: Superior precision for small objects, better discrimination of morphologically similar artifacts
   - **Recommended for**: Clinical deployment requiring high precision

---

### Pre-trained Model Weights

- **`yolo11n.pt`** - YOLOv11 nano variant weights
- **`yolov8n.pt`** - YOLOv8 nano variant weights  
- **`yolov8s.pt`** - YOLOv8 small variant weights (used in active learning cycles)
- **`yolov8m.pt`** - YOLOv8 medium variant weights

These are pre-trained weights from COCO dataset used for transfer learning initialization.

---

## 🔬 Methodology Summary

### Active Learning Pipeline

The system follows an iterative workflow:

1. **Initial Training**: Train YOLOv8 on annotated bacteria + provisional negatives
2. **Deployment**: Generate predictions on unannotated patches (~2% of total WSI patches)
3. **Expert Verification**: Pathologist classifies predictions into:
   - True Positives (TP): Verified bacteria → added to training set
   - False Positives (FP): Hard negatives → added as negative examples
4. **Dataset Enrichment**: Create new dataset combining original + newly verified samples
5. **Iterate**: Train next model and repeat until clinically acceptable performance

### Why YOLOv8 for Active Learning?

- **Speed**: Rapid inference enables processing thousands of patches per iteration
- **Real-time Performance**: Single-stage detector predicts bounding boxes in one forward pass
- **Sufficient Accuracy**: Maintains competitive accuracy for iterative refinement

### Why Faster R-CNN for Final Deployment?

- **Precision**: Two-stage architecture (RPN + classification head) provides superior small object detection
- **Discrimination**: Separate proposal and classification stages better distinguish bacteria from artifacts
- **Clinical Reliability**: Higher confidence predictions reduce false alarms for pathologists

---

## 📊 Key Metrics

### Impact Precision (Clinical Metric)

Standard metrics (mAP, precision, recall) assume complete ground truth annotations. In diagnostic settings, pathologists label only enough bacteria to confirm infection, leaving hundreds unmarked. This makes traditional metrics unreliable.

**Impact Precision** is defined as:

```
P_impact = (Expert-confirmed bacteria) / (Total AI predictions)
```

**Clinical Interpretation**: If Impact Precision = 79%, then when the AI flags a region, there's a 79% probability a pathologist will confirm *H. pylori* presence. This metric determines whether AI assistance reduces or increases pathologist workload.

### Performance Evolution

| Iteration | Dataset Size | Positives | Hard Negatives | Impact Precision | Key Achievement |
|-----------|--------------|-----------|----------------|------------------|-----------------|
| **AL 0** | 1,153 | 1,153 | 0 | 29.6% | Discovered ink artifacts |
| **AL 1** | 4,522 | 2,150 | 2,372 | 59.4% | Learned artifact discrimination |
| **AL 2** | 6,120 | 2,412 | 3,708 | 14.5% | Ratio failure revealed need for balanced negatives |
| **AL 3 (YOLOv8)** | 6,317 | 2,609 | 3,708 | 66.8% | Optimal YOLOv8 configuration |
| **Faster R-CNN** | 6,317 | 2,609 | 3,708 | **79.3%** @ 0.75 | Final deployment model |

---

## 🚀 Usage

### Training Requirements

- **Hardware**: NVIDIA RTX 3090 or equivalent
- **Framework**: PyTorch with Ultralytics (YOLOv8) and torchvision (Faster R-CNN)
- **Dependencies**: Listed in `model_tryouts/requirements.txt`

### Training Configuration

#### YOLOv8 (Active Learning Cycles)
```python
# Training hyperparameters
epochs = 150
batch_size = 16
optimizer = AdamW
learning_rate = 0.001 → 0.0001 (decay)
confidence_threshold = 0.15 (inference, for high recall)
nms_iou = 0.45
```

#### Augmentations (Both Models)
```python
# Orientation invariance
rotation = ±180°
flip = horizontal + vertical

# Staining variations (HSV jitter)
hue = ±0.015
saturation = ±0.7
value = ±0.4

# Scale variations (depth)
scale = 0.5× to 1.5×
```

#### Faster R-CNN (Final Model)
```python
# Architecture
backbone = ResNet-50 + FPN
anchor_sizes = [4, 8, 16, 32, 64] pixels
confidence_threshold = 0.75 (deployment)
pre_training = ImageNet
```

### Deployment Strategies

#### Comprehensive Screening (High Recall)
- **Model**: YOLOv8 from `train_yolo_it3.ipynb`
- **Confidence Threshold**: 0.25
- **Impact Precision**: 79.9%
- **Use Case**: Screening large cohorts, maximize bacteria detection

#### High-Precision Confirmation (Clinical Deployment)
- **Model**: Faster R-CNN from `Train_RCNN_7_02_it3.ipynb`
- **Confidence Threshold**: 0.75
- **Impact Precision**: 79.3%
- **Recall**: Detects 287 of 398 bacteria
- **Use Case**: Clinical diagnostics requiring minimal false alarms

---

## 📈 Dataset Evolution

### The Hard Negative Mining Strategy

Rather than discarding false positives from each iteration, we used them as **hard negatives** - systematically difficult cases where visual similarity to bacteria is highest. These examples provide maximal information gain for learning discriminative boundaries.

#### Iteration 0 → 1: Ink Artifact Discovery
- **Problem**: Model detected dark, curved structures without biological context
- **Solution**: 2,372 ink artifacts became hard negatives
- **Result**: Model learned "what bacteria are NOT"

#### Iteration 1 → 2: Conservative Behavior
- **Problem**: 1:1 ratio made model too cautious
- **Attempted Fix**: 4:1 positive-heavy ratio to recover recall
- **Outcome**: Model memorized specific artifacts without generalizing

#### Iteration 2 → 3: Optimal Balance
- **Solution**: Return to balanced dataset with comprehensive hard negatives
- **Final Dataset**: 2,609 positives + 3,708 diverse hard negatives
- **Coverage**: Ink artifacts, tissue folds, cellular debris, staining irregularities

---

## 🔍 Preprocessing Pipeline

Before model training, WSIs undergo hotspot extraction to focus on diagnostically relevant regions:

1. **Glandular Segmentation**
   - Random Forest pixel classifier (trained in QuPath)
   - Classes: Glandular Epithelium, Stroma, Background
   - *H. pylori* colonizes mucus layer and gastric gland surfaces → glandular regions = hotspots

2. **Patch Extraction**
   - Sliding window: 512×512 pixels, 10% overlap
   - Threshold: Minimum 20% hotspot coverage per patch
   - Efficiency: 60% reduction in search space (~40,000 → ~11,000 patches per slide)

This preprocessing is critical - it directs the detector to focus on morphological distinctions within the mucosal environment rather than learning irrelevant tissue backgrounds.

---

## 📖 References

This work is based on the paper:
> **"From Gigapixels to Bacteria: An Active Learning System for *Helicobacter pylori* Detection in Whole Slide Images"**  
> Authors: Harsha Sathish, Rohan Sanjay Patil, Vidya Padmanabha  
> THWS, MAI, Würzburg, Germany, February 2026

### Key Concepts

- **Active Learning**: Yang et al., "Guided soft attention network for classification of breast cancer histopathology images" (2020)
- **Hard Negative Mining**: Shrivastava et al., "Training region-based object detectors with online hard example mining" (2016)
- **YOLOv8**: Jocher et al., Ultralytics YOLOv8 (2023)
- **Incomplete Annotations**: Bilen & Vedaldi, "Weakly supervised deep detection networks" (2016)

---

## 🎯 Key Findings

1. **Data Quality > Model Complexity**: Clean, verified labels with comprehensive hard negatives improved performance more than architectural innovations

2. **Metric Paradox**: Traditional mAP increased only 7 points (0.41 → 0.48) but clinical utility doubled, demonstrating that standard metrics are unreliable under incomplete annotations

3. **Hard Negative Importance**: Balanced positive-to-negative ratio (1:1) is critical. Insufficient negatives (4:1 ratio) caused model to memorize specific artifacts without generalizing

4. **Two-Stage Superiority**: Faster R-CNN's separate proposal and classification stages excel at discriminating bacteria from morphologically similar artifacts (79% vs 67% precision)

5. **Iterative Discovery**: Active learning discovered 1,456 previously unmarked bacteria, revealing that "complete" expert annotations captured only ~40% of actual bacteria

---

## 🔮 Future Work

- **Multi-stain Support**: Extend to Giemsa, Warthin-Starry, and other staining protocols
- **Laboratory Integration**: Interface with LIS/LIMS for automated report generation
- **Prospective Validation**: Clinical trial comparing AI-assisted vs traditional diagnostic workflows
- **Continued Iteration**: Active learning pipeline can be further refined to improve precision and recall

---

## 🙏 Acknowledgments

- **LABOKLIN**: Provided veterinary gastric biopsy dataset and expert pathological annotations
- **Philipp Ockermann**: Critical support during active learning process
- **Prof. Dr. Magda Gregorová**: Project supervision and guidance
- **THWS**: Computational resources and research infrastructure

---

## 📞 Contact

For questions or collaboration inquiries, please refer to the main repository:  
**GitHub**: [thws-mai/PM25_Helicobacter](https://github.com/thws-mai/PM25_Helicobacter)

---

## 📝 License

This project was developed as part of the MAI Project Module at THWS (Technical University of Applied Sciences Würzburg-Schweinfurt).

---

*Last Updated: February 2026*
