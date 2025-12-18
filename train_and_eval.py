import os
import pandas as pd
import numpy as np
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)

# =========================
# 0. SETUP
# =========================
os.makedirs("results", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# 1. LOAD DATASET
# =========================
df = pd.read_csv("data.csv", encoding="latin-1", engine="python")

df = df[["Tweet", "HS"]]
df = df.rename(columns={"Tweet": "text", "HS": "labels"})

def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["text"] = df["text"].astype(str).apply(clean_text)

print("Total samples:", len(df))

# =========================
# 2. VISUALISASI LABEL DISTRIBUTION
# =========================
plt.figure(figsize=(6,4))
sns.countplot(x=df["labels"])
plt.title("Label Distribution (HS vs Non-HS)")
plt.xlabel("Label (0=Non-HS, 1=HS)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("results/label_distribution.png", dpi=300)
plt.show()

# =========================
# 3. SPLIT DATASET
# =========================
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

train_ds = dataset["train"]
eval_ds = dataset["test"]

# =========================
# 4. VISUALISASI PANJANG TEKS
# =========================
df["text_length"] = df["text"].apply(lambda x: len(x.split()))

plt.figure(figsize=(7,5))
plt.hist(df["text_length"], bins=50)
plt.title("Text Length Distribution")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/text_length_distribution.png", dpi=300)
plt.show()

# =========================
# 5. LOAD INDO-BERT
# =========================
MODEL = "indobenchmark/indobert-base-p1"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2
).to(device)

# =========================
# 6. TOKENIZATION
# =========================
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)

train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
eval_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# =========================
# 7. METRICS
# =========================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }

# =========================
# 8. TRAINING ARGUMENTS
# =========================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,
    report_to="none"
)

# =========================
# 9. TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# =========================
# 10. TRAINING
# =========================
trainer.train()

# =========================
# 11. VISUALISASI LOSS
# =========================
log_history = trainer.state.log_history

train_loss = []
eval_loss = []
epochs = []

for log in log_history:
    if "loss" in log and "epoch" in log:
        train_loss.append(log["loss"])
        epochs.append(log["epoch"])
    if "eval_loss" in log:
        eval_loss.append(log["eval_loss"])

plt.figure(figsize=(7,5))
plt.plot(epochs[:len(train_loss)], train_loss, label="Training Loss")
plt.plot(range(1, len(eval_loss)+1), eval_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=300)
plt.show()

# =========================
# 12. EVALUATION
# =========================
preds = trainer.predict(eval_ds)

y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print(classification_report(y_true, y_pred, target_names=["Non-HS", "HS"]))

# =========================
# 13. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-HS", "HS"],
    yticklabels=["Non-HS", "HS"]
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – IndoBERT Hate Speech")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()

# =========================
# 14. ERROR ANALYSIS SAMPLE
# =========================
eval_df = eval_ds.to_pandas()
eval_df["true"] = y_true
eval_df["pred"] = y_pred

errors = eval_df[eval_df["true"] != eval_df["pred"]]

print("\n=== SAMPLE MISCLASSIFICATIONS ===")
print(errors.sample(5, random_state=42)[["text", "true", "pred"]])

print("\nAll visualizations saved in folder: results/")
