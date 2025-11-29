import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

# -----------------------------
# 1️⃣ Load data
# -----------------------------
data = pd.read_csv("train.csv")
label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

test_texts = data['comment_text'][4000:]
test_labels = data[label_cols].iloc[4000:].reset_index(drop=True)

# -----------------------------
# 2️⃣ Dataset class
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
eval_loader = DataLoader(eval_dataset, batch_size=64)

# -----------------------------
# 5️⃣ Evaluation loop
# -----------------------------
all_probs, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(eval_loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        probs = torch.sigmoid(outputs.logits)
        all_probs.append(probs.cpu())
        all_labels.append(batch['labels'].cpu())

all_probs = torch.cat(all_probs, dim=0).numpy()
all_labels = torch.cat(all_labels, dim=0).numpy()
all_preds = (all_probs > 0.5).astype(int)

# -----------------------------
# 6️⃣ Save results
# -----------------------------
np.save("probs.npy", all_probs)
np.save("labels.npy", all_labels)
np.save("preds.npy", all_preds)

# -----------------------------
# 7️⃣ Compute metrics
# -----------------------------
accuracy = accuracy_score(all_labels, all_preds)
micro_f1 = f1_score(all_labels, all_preds, average='micro')

print(f"✅ Evaluation Complete")
print(f"Accuracy: {accuracy:.4f}")
print(f"Micro F1: {micro_f1:.4f}")

# Per-label performance
prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
for label, p, r, f in zip(label_cols, prec, rec, f1):
    print(f"{label:15s} | Precision: {p:.3f} | Recall: {r:.3f} | F1: {f:.3f}")
