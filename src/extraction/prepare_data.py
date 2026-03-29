# prepare_data.py
from transformers import AutoTokenizer
from preprocess import preprocess_dataset
from datasets import Dataset

# Full BIO label list
LABEL_LIST = [
    "O",
    "B-TOTAL_AMOUNT", "I-TOTAL_AMOUNT",
    "B-DATE", "I-DATE",
    "B-VENDOR", "I-VENDOR"
]

label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for label, i in label2id.items()}


def tokenize_and_align_labels(example, tokenizer, max_length=128):
    """
    Tokenize words and align word-level labels to token-level labels for NER.
    """
    tokenized = tokenizer(
        example["words"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length
    )

    word_ids = tokenized.word_ids()
    labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # ignored in loss
        elif word_idx != previous_word_idx:
            labels.append(label2id[example["labels"][word_idx]])
        else:
            labels.append(-100)  # subword continuation
        previous_word_idx = word_idx

    tokenized["labels"] = labels
    return tokenized


def prepare_data(max_examples=None):
    """
    Load processed data, tokenize, align labels, and return HF Dataset.
    """
    print("Loading processed dataset...")
    data = preprocess_dataset()

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    print("Tokenizing and aligning labels...")

    tokenized_data = []
    if max_examples:
        data = data[:max_examples]

    for example in data:
        tokenized = tokenize_and_align_labels(example, tokenizer)
        tokenized_data.append(tokenized)

    # Convert list of dicts to HF Dataset
    hf_dataset = Dataset.from_list(tokenized_data)
    print(f"Prepared dataset with {len(hf_dataset)} examples.")

    return hf_dataset, tokenizer


if __name__ == "__main__":
    dataset, tokenizer = prepare_data(max_examples=500)
    print("\nSample tokenized dataset entry:\n")
    print(dataset[0])