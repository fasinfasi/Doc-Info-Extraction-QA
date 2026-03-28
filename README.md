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