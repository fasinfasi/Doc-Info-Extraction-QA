from fastapi import FastAPI
from pydantic import BaseModel

from src.extraction.inference import load_model, predict, extract_entities

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

# Root
@app.get("/")
def root():
    return {"message": "API is running"}

@app.post("/extract")
def extract(request: ExtractRequest):
    preds = predict(request.text, tokenizer, model)
    structured = extract_entities(preds)

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

    if "total" in question:
        answer = structured["total_amount"]
    elif "date" in question:
        answer = structured["date"]
    elif "vendor" in question or "store" in question:
        answer = structured["vendor"]
    else:
        answer = "Answer not found"

    return {
        "status": "success",
        "answer": answer
    }