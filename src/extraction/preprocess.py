# preprocess.py
from datasets import load_dataset

# Full BIO label mapping for NER
ENTITY_LABELS = ["TOTAL_AMOUNT", "DATE", "VENDOR"]

def create_ner_example(example):
    """
    Convert raw dataset example to NER example with BIO labels.
    """
    words = example["words"]
    entities = example["entities"]

    labels = ["O"] * len(words)

    def assign_label(entity_value, label_name):
        if entity_value is None:
            return
        tokens = entity_value.split()
        for i in range(len(words)):
            if words[i:i+len(tokens)] == tokens:
                labels[i] = f"B-{label_name}"
                for k in range(1, len(tokens)):
                    labels[i+k] = f"I-{label_name}"

    assign_label(entities.get("company"), "VENDOR")
    assign_label(entities.get("date"), "DATE")
    assign_label(entities.get("total"), "TOTAL_AMOUNT")

    return {
        "words": words,
        "labels": labels
    }


def preprocess_dataset():
    """
    Load dataset from Hugging Face and preprocess all examples.
    Returns a list of dicts: {"words": [...], "labels": [...]}
    """
    print("Loading dataset from Hugging Face...")
    dataset = load_dataset("jsdnrs/ICDAR2019-SROIE")["train"]

    print("Processing examples...")
    processed_data = [create_ner_example(x) for x in dataset]

    print(f"Processed {len(processed_data)} examples.")
    return processed_data


if __name__ == "__main__":
    data = preprocess_dataset()
    print("\nSample processed example:\n")
    print(data[0])