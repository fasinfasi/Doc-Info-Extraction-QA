# Document Information Extraction & QA System

## Project Overview
This project extracts structured information from receipts and enables question answering.

## Setup Instructions

```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```

## Dataset Loading

We use the Voxel51 CORD dataset.

To load dataset:

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub("Voxel51/consolidated_receipt_dataset")

## Data Preprocessing

Converted dataset into NER format:
- Extracted words from OCR
- Mapped labels to:
  - TOTAL_AMOUNT
  - DATE
  - VENDOR
  - O (others)

## Data Preparation

- Tokenized text using BERT tokenizer
- Aligned labels with tokens
- Converted labels to IDs for training

## Model Training

- Fine-tuned BERT for token classification
- Labels:
  - TOTAL_AMOUNT
  - DATE
  - VENDOR
  - O

**To train in local**:
```bash
python src/extraction/train.py
```

NOTE: The NER model was trained using Google Colab to leverage GPU acceleration.
The final trained model was exported and integrated into the local project environment developed in VS Code. Code placed in `notebooks/training_colab.ipynb`



