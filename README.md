# Document Information Extraction & QA System📃

## Project Overview
This project extracts structured information from receipt using Named Entity Recognition (NER) and provides a simple Question Answering (QA) interface.

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

```bash
python -m venv venv
source venv/Scripts/activate
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

## API Development
Built using FastAPI.

## Run API
```bash
uvicorn src.api.main:app --reload
```

### Endpoints
```/extract```
Extract structured data from document.

**Request**:
```json
{
  "text": "KFC 2023-12-05 Total 45.99"
}
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
├── notebooks
│   └── training_colab.ipynb
├── src
│   ├── api
│   │   └── main.py
│   ├── extraction
│   │   ├── inference.py
│   │   ├── prepare_data.py
│   │   ├── preprocess.py
│   │   └── train.py
│   ├── qa
│   └── utils
│       └── load_extraction.py
├── ui
├── .gitignore
├── LICENSE
├── README.md
├── app.py
└── requirements.txt
```

### Notes
- The model may predict only 'O' labels due to:
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

