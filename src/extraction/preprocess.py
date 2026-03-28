import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub


# Step 1: Label mapping
LABEL_MAP = {
    "total.total_price": "TOTAL_AMOUNT",
    "date": "DATE",
    "store_info.name": "VENDOR",
}


def map_label(original_label):
    return LABEL_MAP.get(original_label, "O")


def preprocess_dataset():
    print("Loading dataset...")
    dataset = load_from_hub("Voxel51/consolidated_receipt_dataset")

    processed_data = []

    print("Processing samples...")

    for sample in dataset:
        words = []
        labels = []

        for det in sample.detections.detections:
            word = det["text"]
            label = map_label(det.label)

            words.append(word)
            labels.append(label)

        processed_data.append({
            "words": words,
            "labels": labels
        })

    print("Preprocessing complete!")
    return processed_data


if __name__ == "__main__":
    data = preprocess_dataset()

    # Print one processed sample
    print("\nSample processed output:\n")
    print(data[0])