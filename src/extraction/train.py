# train.py
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from prepare_data import prepare_data, label2id, id2label

def train_model():
    print("Preparing dataset...")
    train_dataset, tokenizer = prepare_data(max_examples=500)

    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        "dslim/bert-base-NER",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    training_args = TrainingArguments(
        output_dir="ner-model",
        learning_rate=3e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        logging_dir="logs"
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    print("Starting training...")
    trainer.train()

    print("Saving model and tokenizer...")
    trainer.save_model("ner-model")
    tokenizer.save_pretrained("ner-model")
    print("Training complete!")


if __name__ == "__main__":
    train_model()