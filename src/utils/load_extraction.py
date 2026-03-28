import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub


def load_dataset():
    print("Loading Voxel51 CORD dataset...")

    dataset = load_from_hub("Voxel51/consolidated_receipt_dataset")

    print("\nDataset Loaded Successfully!")
    print(f"Number of samples: {len(dataset)}")

    return dataset


if __name__ == "__main__":
    dataset = load_dataset()

    # Inspect one sample
    sample = dataset.first()

    print("\nSample fields:")
    print(sample)

    print("\nDetections (OCR + labels):")
    for det in sample.detections.detections[:5]:
        print(f"Text: {det.label}, OCR: {det['text']}")