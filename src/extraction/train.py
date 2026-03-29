
# Local version training model

from transformers import (
    BertForTokenClassification,
    TrainingArguments,
    Trainer
)
from prepare_data import prepare_data, label2id, id2label


def train_model():
    print("Preparing dataset...")
    dataset = prepare_data()

    print("Loading model...")
    model = BertForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir="models/",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=10,
        save_strategy="epoch"
    )

    print("Starting training...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    trainer.train()

    print("Saving model...")
    model.save_pretrained("models/ner-model")

    print("Training complete!")


if __name__ == "__main__":
    train_model()