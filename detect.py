import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =============================
# CONFIG
# =============================
MODEL_PATH = "./results/checkpoint-1977"   # folder hasil training
LABEL_MAP = {0: "Non-HS", 1: "HS"}

# =============================
# CLEANING (SAMA DENGAN TRAINING)
# =============================
def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+", "", t)
    t = re.sub(r"@\w+", "", t)
    t = re.sub(r"#\w+", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# =============================
# LOAD MODEL
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# =============================
# PREDICTION FUNCTION
# =============================
def detect_hate_speech(text):
    text = clean_text(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)

    label_id = torch.argmax(probs, dim=1).item()
    confidence = probs[0][label_id].item()

    return {
        "text": text,
        "label": LABEL_MAP[label_id],
        "confidence": round(confidence, 4)
    }

# =============================
# INTERACTIVE MODE
# =============================
if __name__ == "__main__":
    print("\nHate Speech Detection (ketik 'exit' untuk keluar)\n")

    while True:
        user_input = input("Text > ")
        if user_input.lower() == "exit":
            break

        result = detect_hate_speech(user_input)
        print("Prediction:", result["label"])
        print("Confidence:", result["confidence"])
        print("-" * 40)
