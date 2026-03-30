# Document Information Extraction & QA System📃

## Project Overview
This project extracts structured information from receipt using Named Entity Recognition(NER) + OCR extraction and provides a simple Question Answering (QA) interface.

The system identifies key entities:
- Total Amount
- Date
- Vendor

It includes:
- Model training (BERT-based NER)
- REST API (FastAPI)
- User Interface (Streamlit)

---

## Setup Instructions

1. Clone the Repository
```bash
git clone https://github.com/fasinfasi/Doc-Info-Extraction-QA.git
cd Doc-Info-Extraction-QA
```

2. Create Environment
```bash
python -m venv venv
```

3. Activate Virtual Environment
- Windows
```bash
venv/Scripts/activate
```
- Mac/Linux
```bash
source venv/Scripts/activate
```

4. Install Dependencies
```bash
pip install -r requirements.txt
```

## Dataset
Used the ICDAR 2019 SROIE (Scanned Receipts OCR and Information Extraction) dataset.

**To Load Dataset:**
```python
from datasets import load_dataset
dataset = load_dataset("jsdnrs/ICDAR2019-SROIE")
```

## Data Preprocessing
Converted dataset into NER Format:
- Extraction OCR words
- Mapped labels:
  - TOTAL_AMOUNT
  - DATE
  - VENDOR
  - O (Others)

Used BIO tagging format:
- B = (Beginning)
- I = (Inside)
- O = (Outside)

## Data Preparation
- Tokenization using pretrained BERT tokenizer (dslim/bert-base-NER)
- Word-to-token alignment using word_ids()
- Ignored subword tokens using -100
- Max sequence length: 128

## Model Training
Fine-tuned a transformer model for token classification.

**Base Model**:
- dslim/bert-base-NER

**Key Notes**:
- Classifier head reinitialized due to label mismatch
- Used ignore_mismatched_sizes=True

**Training executed on**:
- Google Colab (GPU)

**Notebook**:
```bash
notebooks/training_colab.ipynb
```

## Model Inference
- Token-level predictions generated from trained model
- Converted into structured output:

```json
{
  "total_amount": ...,
  "date": ...,
  "vendor": ...
}
```

## OCR Integration
The system uses OCR to extract text from receipt images before passing it to the NER model.

- Input: Receipt image (PNG/JPG/JPEG)
- OCR Engine: Tesseract
- Output: Raw extracted text

This enables real-world usage where users upload receipt images instead of manually entering text.

## API Development
Built using FastAPI.

## Run API
```bash
uvicorn src.api.main:app --reload
```

### Endpoints
```/extract```

Extract structured data from a receipt image.

**Request**:
- Type: `multipart/form-data`
- Field: `file`

Upload a receipt image file.

**Example (cURL)**:
```bash
curl -X POST "http://127.0.0.1:8000/extract" \
  -F "file=@receipt.jpg"
```

**Response**:
```json
{
  "status": "success",
  "data": {
    "total_amount": null,
    "date": null,
    "vendor": null
  }
}
```

```/query```

Answer questions based on extracted data

**Request**
```json
{
  "text": "KFC 2023-12-05 Total 45.99",
  "question": "What is the total amount?"
}
```

## Data Storage
Store every extracted data to `data/extracted.json`

Store Data Include:
```json
{
  "timestamp": "...",
  "input_text": "...",
  "predictions": [...],
  "structured_data": {...}
}
```

## User Interface
Build using Streamlit

### Run UI
```bash
streamlit run app.py
```

### Features
- Input receipt text
- View extracted structured data
- View token predictions
- Ask questions interactively

## System Architecture
```
[ Streamlit UI ]
        ↓
[ FastAPI Backend ]
        ↓
[ NER Model (BERT) ]
```

- Streamlit provides user interface
- FastAPI handles requests and responses
- Model performs entity extraction
This architecture is commonly used for rapid AI application development

## Project Structure
```
├── data
│   └── extracted.json    # Store Extracted Data
├── notebooks
│   └── training_colab.ipynb    # Model Build
├── models
│   └── ner-model       # NER model
├── src
│   ├── api
│   │   └── main.py   # FastAPI
│   ├── extraction
│   │   ├── inference.py
│   │   ├── prepare_data.py
│   │   ├── preprocess.py
│   │   └── train.py
│   ├── utils
│   │   ├── load_extraction.py
│   │   └── ocr.py
│   └── config.py
├── .gitignore
├── LICENSE
├── README.md
├── app.py      # UI
└── requirements.txt

```

### Notes
- The model may predict only 'O'(others) labels due to:
  - Limited dataset size
  - Class imbalance
  - Domain adaptation challenges
- Despite this, the system successfully demonstrates:
  - End-to-end ML pipeline
  - Model training and inference
  - API deployment
  - Interactive UI

## Conclusion
This project demonstrates a complete pipeline for document information extraction and question answering using modern NLP tools and web frameworks.

