# ML Reporting Checklist

## Module 1: Study Goals

**1a. Population**
~12,500 augmented microscopy images of four white blood cell (WBC) subtypes: Eosinophil, Lymphocyte, Monocyte, and Neutrophil. The original 410 images are excluded to avoid data leakage from augmentation overlap across splits.

**1b. Motivation**
WBC subtype identification is a routine step in diagnosing blood-based diseases (e.g., leukemia, anemia, immune disorders). Manual classification requires trained specialists and is time-intensive, making it a natural target for automation.

**1c. Motivation for ML**
The task is image classification — distinguishing cell types based on visual morphology. CNNs are well-suited for this because they learn spatial features (shape, texture, color) directly from raw pixels, without requiring hand-crafted features. Transfer learning from ImageNet-pretrained models further reduces the data requirement, which is relevant given the small original dataset (410 images).

---

## Module 2: Computational Reproducibility

**2a. Dataset**
Kaggle Blood Cell Images by Paul Mooney. [https://www.kaggle.com/datasets/paultimothymooney/blood-cells](https://www.kaggle.com/datasets/paultimothymooney/blood-cells). License: MIT.

**2b. Code**
[https://github.com/marina15rodriguez/bloodcells](https://github.com/marina15rodriguez/bloodcells)

**2c. Computing infrastructure**
- Training: Kaggle Notebook with GPU (NVIDIA T4/P100)
- OS: Linux (Kaggle environment)
- Python 3.10+
- Dependencies: see `requirements.txt`

**2d. README**
See `README.md` in the repository root.

**2e. Reproduction script**
```bash
git clone https://github.com/marina15rodriguez/bloodcells.git
cd bloodcells
pip install -r requirements.txt
python src/train.py --data-dir data/ --epochs 20 --warmup-epochs 5 --output-dir results/
python src/evaluate.py --checkpoint results/best_model.pth --data-dir data/
```

---

## Module 3: Data Quality

**3a. Data source**
Kaggle Blood Cell Images dataset by Paul Mooney, derived from the BCCD Dataset. Contains augmented microscopy images of WBCs with class labels. Only `dataset2-master` (augmented set) is used.

**3b. Sampling frame**
Microscopy images of peripheral blood smears, stained and cropped to individual cells.

**3c. Justification**
The dataset contains labeled examples of the exact 4 WBC subtypes the model is designed to classify, at sufficient scale (~12,500 images) for transfer learning.

**3d. Outcome variable**
WBC subtype — a 4-class categorical variable: Eosinophil, Lymphocyte, Monocyte, Neutrophil.

**3e. Number of samples**

| Split | Eosinophil | Lymphocyte | Monocyte | Neutrophil | Total |
|-------|-----------|-----------|---------|-----------|-------|
| TRAIN | 2497 | 2483 | 2478 | 2499 | 9957 |
| TEST  | 623  | 620  | 620  | 624  | 2487 |

**3f. Missing data**
None. All 4 classes are present in both splits with no missing labels.

**3g. Representativeness**
The dataset represents augmented images from a single source with one staining protocol. Generalizability to other labs or protocols is limited (see Module 8).

---

## Module 4: Data Preprocessing

**4a. Excluded samples**
`dataset-master` (410 original images) excluded to avoid leakage from augmentation overlap with `dataset2-master`.

`TEST_SIMPLE` (71 images) excluded — too small for meaningful evaluation.

**4b. Corrupt samples**
None encountered.

**4c. Transformations**

Training pipeline:
1. Resize to 240×240
2. Random horizontal flip
3. Random vertical flip
4. Random rotation (±15°)
5. Color jitter (brightness, contrast, saturation ±0.2)
6. Convert to tensor
7. Normalize with ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`

Validation/Test pipeline:
1. Resize to 240×240
2. Convert to tensor
3. Normalize with ImageNet mean and std

ImageNet statistics are used because the model backbone was pretrained on ImageNet.

---

## Module 5: Modeling

**5a. Model description**
- **Architecture**: EfficientNet-B0 (via `timm`) with a Linear(1280, 4) classification head
- **Features**: Raw RGB pixels (240×240)
- **Loss function**: Cross-entropy loss

**5b. Justification**
EfficientNet-B0 achieves high accuracy with relatively few parameters, reducing overfitting risk on a small dataset. ImageNet pretraining provides strong general visual features transferable to microscopy images.

**5c. Train/test split**
- TRAIN (9,957 images) split 80/20 into train/val using fixed seed (42)
- TEST (2,487 images) held out for final evaluation

**5d. Model selection**
Checkpoint with highest validation accuracy saved during training.

**5e. Hyperparameters**
| Parameter | Value |
|-----------|-------|
| Epochs | 20 (5 warm-up + 15 fine-tune) |
| Batch size | 32 |
| Learning rate | 1e-3 (head), 1e-4 (backbone during fine-tune) |
| Weight decay | 1e-4 |
| LR scheduler | CosineAnnealingLR |
| Seed | 42 |

**5f. Baselines**
No explicit baseline comparison. A random classifier would achieve 25% accuracy (balanced 4-class problem).

---

## Module 6: Data Leakage

**6a. Preprocessing uses only training data**
Normalization uses ImageNet statistics (computed from a separate dataset), not test set statistics. No leakage.

**6b. Train/test dependencies**
The pre-existing TRAIN/TEST split from Kaggle is used as-is. The original 410 images and their augmentations are kept within the same split by excluding `dataset-master` entirely.

**6c. Feature legitimacy**
Raw pixel values are the only input features. No engineered features that could encode label information.

---

## Module 7: Metrics and Uncertainty

**7a. Metrics**
Per-class precision, recall, F1-score, and overall accuracy reported on the held-out TEST set.

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Eosinophil | 0.908 | 0.867 | 0.887 |
| Lymphocyte | 0.998 | 0.979 | 0.989 |
| Monocyte | 0.996 | 0.760 | 0.862 |
| Neutrophil | 0.705 | 0.917 | 0.797 |
| **Overall** | **0.901** | **0.881** | **0.884** |

**7b. Uncertainty estimates**
No confidence intervals reported. A single train/test split was used; results may vary across different splits.

**7c. Statistical tests**
None performed. Single dataset evaluation only.

---

## Module 8: Generalizability and Limitations

**8a. External validity**
Limited. The model was trained and evaluated on a single dataset (Kaggle blood-cells by Paul Mooney), consisting of augmented images from 410 original microscopy photos. Performance on images from different labs, staining protocols, or microscope settings is unknown.

**8b. Contexts where findings may not hold**
- **Different staining protocols** — WBC appearance varies with staining technique. The model learned features specific to the staining used in this dataset.
- **Monocyte/Neutrophil confusion** — the model struggles to distinguish these two classes (Monocyte recall = 0.76, Neutrophil precision = 0.71). Clinical use for these subtypes would require further improvement.
- **Image quality variation** — all images are clean, well-centered microscopy crops. Performance on noisier or differently framed images is unknown.
- **Rare cell morphologies** — abnormal cell appearances (e.g. in disease states like leukemia) were not present in training data. The model may misclassify pathological cells.

## Potential Improvements

**Data**
- Collect more diverse examples of Monocyte and Neutrophil, the two weakest classes.
- Apply stronger augmentation (elastic distortion, random erasing) for better robustness.

**Training**
- Increase warm-up epochs (e.g. 10) to give the head more time to stabilize before unfreezing.
- Use label smoothing to prevent overconfidence and improve generalization.
- Apply weighted loss to upweight Monocyte and Neutrophil during training.

**Model**
- Try a larger EfficientNet variant (B2 or B3) for more capacity.
- Apply test-time augmentation (TTA) — average predictions over multiple augmented views of each test image for a free accuracy boost without retraining.

**Evaluation**
- Use k-fold cross-validation to obtain more reliable performance estimates and confidence intervals (addresses the uncertainty gap in Module 7b).
