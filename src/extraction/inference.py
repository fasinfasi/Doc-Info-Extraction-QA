from transformers import AutoTokenizer, BertForTokenClassification
import torch
import os
import sys

MODEL_PATH = "models/ner-model"

def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = BertForTokenClassification.from_pretrained(MODEL_PATH)
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)


def predict(text, tokenizer, model):
    words = text.split()
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    predicted_labels = []
    word_ids = inputs.word_ids()  # maps tokens to original words

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        label_id = predictions[0][idx].item()
        label = model.config.id2label[label_id]
        predicted_labels.append((words[word_idx], label))

    return predicted_labels

def extract_entities(predictions):
    result = {
        "total_amount": None,
        "date": None,
        "vendor": None
    }

    for word, label in predictions:
        if label == "TOTAL_AMOUNT" and result["total_amount"] is None:
            result["total_amount"] = word
        elif label == "DATE" and result["date"] is None:
            result["date"] = word
        elif label == "VENDOR" and result["vendor"] is None:
            result["vendor"] = word

    return result

if __name__ == "__main__":
    tokenizer, model = load_model()

    # Example text
    test_text = "Total 45.99 KFC 2023-12-05"

    # Get predictions
    preds = predict(test_text, tokenizer, model)
    print("\nPredictions:\n", preds)

    # Get structured output
    structured_output = extract_entities(preds)
    print("\nStructured Output:\n", structured_output)