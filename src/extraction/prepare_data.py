from transformers import AutoTokenizer
from preprocess import preprocess_dataset


# Label list
LABEL_LIST = ["O", "TOTAL_AMOUNT", "DATE", "VENDOR"]

label2id = {label: i for i, label in enumerate(LABEL_LIST)}
id2label = {i: label for label, i in label2id.items()}


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["words"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    word_ids = tokenized_inputs.word_ids()

    labels = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)  # ignore
        elif word_idx != previous_word_idx:
            labels.append(label2id[examples["labels"][word_idx]])
        else:
            labels.append(label2id[examples["labels"][word_idx]])

        previous_word_idx = word_idx

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def prepare_data():
    print("Loading processed data...")
    data = preprocess_dataset()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    print("Tokenizing data...")

    tokenized_data = []

    for sample in data[:200]:  # limit for speed
        tokenized = tokenize_and_align_labels(sample, tokenizer)
        tokenized_data.append(tokenized)

    print("Data preparation complete!")
    return tokenized_data


if __name__ == "__main__":
    dataset = prepare_data()

    print("\nSample tokenized output:\n")
    print(dataset[0])