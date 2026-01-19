import torch
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score
import os
os.environ["WANDB_DISABLED"] = "true"


from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# -----------------------------
# Load IMDB dataset
# -----------------------------
dataset = load_dataset("imdb")

# -----------------------------
# Load tokenizer
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# -----------------------------
# Load model
# -----------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}


# -----------------------------
# Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./bert_model",
    eval_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)



# -----------------------------
# Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics
)


# -----------------------------
# Train
# -----------------------------
trainer.train()

# -----------------------------
# Save model
# -----------------------------
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

print("✅ BERT model saved successfully!")
