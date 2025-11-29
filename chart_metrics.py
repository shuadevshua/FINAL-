import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score
from tqdm import tqdm

# -----------------------------
# 1️⃣ Load data
# -----------------------------
data = pd.read_csv("train.csv")
label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

# Use same split as training
test_texts = data['comment_text'][4000:]
test_labels = data[label_cols].iloc[4000:].reset_index(drop=True)

# -----------------------------
# 2️⃣ Dataset class with pre-tokenization
# -----------------------------
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels.values, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# -----------------------------
# 3️⃣ Load model and tokenizer
# -----------------------------
model_path = "./bert_toxic_model_multilabel_final"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# -----------------------------
# 4️⃣ Prepare dataset and dataloader
# -----------------------------
eval_dataset = ToxicDataset(test_texts, test_labels, tokenizer)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)  # bigger batch = faster

# -----------------------------
# 5️⃣ Evaluation loop
# -----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        probs = torch.sigmoid(outputs.logits)
        all_preds.append((probs > 0.5).cpu())
        all_labels.append(batch['labels'].cpu())

# Concatenate all batches
all_preds = torch.cat(all_preds, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()

# -----------------------------
# 6️⃣ Compute metrics
# -----------------------------
accuracy = accuracy_score(all_labels, all_preds)
micro_f1 = f1_score(all_labels, all_preds, average='micro')
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)

print(f"✅ Evaluation Complete")
print(f"Overall Accuracy: {accuracy:.4f}")
print(f"Overall Micro F1: {micro_f1:.4f}")

# -----------------------------
# 7️⃣ Plot per-label F1 scores
# -----------------------------
plt.figure(figsize=(8, 5))
plt.bar(label_cols, f1, color='skyblue')
plt.title("F1 Score per Toxic Label")
plt.xlabel("Label")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
