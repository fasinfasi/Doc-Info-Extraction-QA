from fastapi import FastAPI
from pydantic import BaseModel

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
def extract(request: ExtractRequest):
    preds = predict(request.text, tokenizer, model)
    structured = extract_entities(preds)

    # Store Extracted Data
    record = {
        "timestamp": datetime.now().isoformat(),
        "input_text": request.text,
        "predictions": preds,
        "structured_data": structured
    }

    try:
        with open(DATA_PATH, "a", encoding="utf-8") as f:
            json.dump(record, f)
            f.write("\n")
    except Exception as e:
        print(f"⚠️ Failed to save data: {e}")

    return {
        "status": "success",
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

    # 🧠 Smart answering logic
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