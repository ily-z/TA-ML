import pandas as pd
import numpy as np
import re
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


#load dataset
df = pd.read_csv("data.csv", encoding="latin-1", engine="python")

df = df[["Tweet", "HS"]]
df = df.rename(columns={
    "Tweet": "text",
    "HS": "labels"
})

def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

df["text"] = df["text"].astype(str).apply(clean_text)

#split dataset

dataset = Dataset.from_pandas(df)

dataset = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_ds = dataset["train"]
eval_ds = dataset["test"]

#load indo bert

MODEL = "indobenchmark/indobert-base-p1"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2
)

#tokonize dataset

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


#metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


#training arguments


training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    #evaluation_strategy="epoch",
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


#trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


#training 
trainer.train()


#evaluation

preds = trainer.predict(eval_ds)

y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=1)

print(classification_report(y_true, y_pred, target_names=["Non-HS", "HS"]))
print(confusion_matrix(y_true, y_pred))
