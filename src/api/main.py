from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import UploadFile, File
from src.utils.ocr import extract_text_from_image

from src.extraction.inference import load_model, predict, extract_entities

import json
import os
from datetime import datetime

app = FastAPI(title="Document Information Extraction API")


# Load model once at startup
tokenizer = None
model = None

@app.on_event("startup")
def startup_event():
    global tokenizer, model
    tokenizer, model = load_model()


class ExtractRequest(BaseModel):
    text: str


class QueryRequest(BaseModel):
    text: str
    question: str

DATA_PATH = "data/extracted.json"
os.makedirs("data", exist_ok=True)


# Root
@app.get("/")
def root():
    return {"message": "API is running"}


@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    # Save uploaded file
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # OCR
    extracted_text = extract_text_from_image(file_path)

    # Model prediction
    preds = predict(extracted_text, tokenizer, model)
    structured = extract_entities(preds)

    # Save record
    record = {
        "timestamp": datetime.now().isoformat(),
        "input_text": extracted_text,
        "predictions": preds,
        "structured_data": structured
    }

    with open(DATA_PATH, "a", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")

    return {
        "status": "success",
        "text": extracted_text,
        "predictions": preds,
        "data": structured
    }


@app.post("/query")
def query(request: QueryRequest):
    preds = predict(request.text, tokenizer, model)
    structured = extract_entities(preds)

    question = request.question.lower()

    # Extract values
    total = structured.get("total_amount")
    date = structured.get("date")
    vendor = structured.get("vendor")

    if "total" in question:
        if total:
            answer = f"The total amount on the receipt is {total}."
        else:
            answer = "I couldn't confidently identify the total amount from the document."

    elif "date" in question:
        if date:
            answer = f"The transaction date appears to be {date}."
        else:
            answer = "I wasn't able to detect a clear date in the document."

    elif "vendor" in question or "store" in question:
        if vendor:
            answer = f"The vendor or store is likely {vendor}."
        else:
            answer = "I couldn't determine the vendor from the provided text."

    else:
        # General fallback
        answer = (
            "I'm not sure how to answer that question. "
            "You can ask about total amount, date, or vendor."
        )

    return {
        "status": "success",
        "answer": answer,
        "extracted_data": structured
    }