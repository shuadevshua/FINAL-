from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

# -------------------------------
# ✅ 1. Locate the model folder
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "bert_toxic_model_multilabel_final")

print(f"[INFO] Loading model from: {MODEL_PATH}")

# -------------------------------
# ✅ 2. Load tokenizer & model once
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
model.eval()
print("[INFO] Model loaded successfully.")

# -------------------------------
# ✅ 3. Define your label names
# -------------------------------
LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# -------------------------------
# ✅ 4. Define the prediction function
# -------------------------------
def predict_comment(comment, threshold=0.5):
    """Predict toxic category probabilities for a given comment (multi-label)."""
    print(f"[DEBUG] Running prediction for: {comment}")

    inputs = tokenizer(comment, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Ensure output is moved to CPU before converting
        probs = torch.sigmoid(outputs.logits)[0].detach().cpu().numpy().tolist()

    results = {label: float(prob) for label, prob in zip(LABELS, probs)}
    print(f"[DEBUG] Prediction results: {results}")
    return results


# -------------------------------
# ✅ 5. Run a standalone test
# -------------------------------
if __name__ == "__main__":
    sample = "I hate you so much!"
    output = predict_comment(sample)
    print(f"\nComment: {sample}\nToxicity breakdown:")
    for label, value in output.items():
        print(f" - {label}: {value:.3f}")
