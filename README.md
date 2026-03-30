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

Used the ICDAR 2019 SROIE (Scanned Receipts OCR and Information Extraction) dataset.

To load dataset:

```python
from datasets import load_dataset
dataset = load_dataset("jsdnrs/ICDAR2019-SROIE")
```

## Data Preprocessing

Converted dataset into NER format:
- Extracted words from OCR
- Mapped labels to:
  - TOTAL_AMOUNT
  - DATE
  - VENDOR
  - O (others)

## Data Preparation

- Tokenization using pretrained BERT tokenizer (dslim/bert-base-NER)
- Word-to-token label alignment
- Word-to-token label alignment

## Model Training

We fine-tune a pretrained transformer model for token classification.

**Base Model**:
  - dslim/bert-base-NER

**Key Note**:
- The classifier head is reinitialized due to label mismatch
- `ignore_mismatched_sizes=True` is used

**To train in local**:
```bash
python src/extraction/train.py
```

NOTE: The NER model was trained using Google Colab to leverage GPU acceleration. The final trained model was exported and integrated into the local project environment developed in VS Code. Code placed in `notebooks/training_colab.ipynb`


Note:
The current model may predict 'O' labels due to limited dataset size and class imbalance.
However, the system successfully demonstrates the complete pipeline from training to API deployment.