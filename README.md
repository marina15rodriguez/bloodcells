# Blood Cell Classification

Automated classification of white blood cells (WBC) using deep learning and computer vision techniques.

## Overview

This project tackles the multi-class image classification of white blood cells into 4 subtypes. Automated WBC classification has direct clinical applications in diagnosing blood-based diseases such as leukemia, anemia, and immune system disorders — tasks traditionally performed manually by trained hematologists.

## Dataset

**Source:** [Blood Cell Images — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data) by Paul Mooney
**License:** MIT

| Split | Images |
|-------|--------|
| Augmented dataset | ~12,500 JPEG images |
| Original dataset | 410 images (with XML bounding boxes) |

### Classes

| Class | Description |
|-------|-------------|
| Eosinophil | Associated with allergic reactions and parasitic infections |
| Lymphocyte | Key players in the adaptive immune response |
| Monocyte | Largest WBC, involved in phagocytosis and inflammation |
| Neutrophil | Most abundant WBC, first responders to infection |

Each class contains approximately 3,000 images in the augmented set (~113 MB total).

## Project Goals

- Build a CNN-based classifier to identify WBC types from microscopy images
- Explore transfer learning with pretrained architectures (e.g., ResNet, EfficientNet)
- Evaluate model performance with accuracy, precision, recall, and F1-score per class
- Visualize predictions and class activation maps (CAM/Grad-CAM)

## Project Structure

```
bloodcells/
├── data/               # Dataset (not tracked in git)
├── notebooks/          # Exploratory data analysis and experiments
├── src/                # Source code
│   ├── dataset.py      # Data loading and augmentation
│   ├── model.py        # Model architecture
│   ├── train.py        # Training loop
│   └── evaluate.py     # Evaluation and metrics
├── results/            # Saved models, plots, and metrics
├── requirements.txt    # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch or TensorFlow
- See `requirements.txt` for full dependencies

### Installation

```bash
git clone https://github.com/marina15rodriguez/bloodcells.git
cd bloodcells
pip install -r requirements.txt
```

### Download the Dataset

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data) and place it under `data/`.

```bash
kaggle datasets download -d paultimothymooney/blood-cells
unzip blood-cells.zip -d data/
```

## Results

_To be updated as experiments progress._

## References

- Original data sourced from the [BCCD Dataset](https://github.com/Shenggan/BCCD_Dataset)
- Paul Mooney, Kaggle Blood Cell Images Dataset
