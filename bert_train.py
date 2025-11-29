import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import transformers

print("Transformers version in use:", transformers.__version__)
print("Transformers file path:", transformers.__file__)

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = pd.read_csv("train.csv")
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# -----------------------------
# 2️⃣ Train / Validation / Test Split (80 / 10 / 10)
# -----------------------------
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    data["comment_text"],
    data[label_cols],
    test_size=0.2,
    random_state=42,
    stratify=data[label_cols].sum(axis=1) > 0
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts,
    temp_labels,
    test_size=0.5,
    random_state=42
)

print(f"✅ Train size: {len(train_texts)}")
print(f"✅ Validation size: {len(val_texts)}")
print(f"✅ Test size: {len(test_texts)}")

# -----------------------------
# 3️⃣ Dataset class
# -----------------------------
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = torch.tensor(self.labels.iloc[idx].values.astype(float))
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze() for k, v in encoding.items()}
        item["labels"] = label
        return item

# -----------------------------
# 4️⃣ Tokenizer and model
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

# -----------------------------
# 5️⃣ Device setup
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple MPS accelerator")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (training will be slower).")
model.to(device)

# -----------------------------
# 6️⃣ Create datasets
# -----------------------------
train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
val_dataset = ToxicDataset(val_texts, val_labels, tokenizer)
test_dataset = ToxicDataset(test_texts, test_labels, tokenizer)

# -----------------------------
# 7️⃣ Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./bert_toxic_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,       # ✅ fits 8GB GPU
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,       # ✅ simulates batch 32
    num_train_epochs=3,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=True,                           # ✅ faster on RTX
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"                     # Disable wandb/logging services
)

# -----------------------------
# 8️⃣ Compute metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()
    labels = torch.tensor(labels).int()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    return {"accuracy": acc, "f1": f1}

# -----------------------------
# 9️⃣ Trainer + Real-time metrics tracker
# -----------------------------
train_f1, val_f1, val_acc = [], [], []

class MetricsCallback(transformers.TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        val_f1.append(metrics["eval_f1"])
        val_acc.append(metrics["eval_accuracy"])
        tqdm.write(f"📊 Epoch {state.epoch:.0f} | F1: {metrics['eval_f1']:.4f} | Acc: {metrics['eval_accuracy']:.4f}")
        plt.clf()
        plt.plot(val_f1, label="Validation F1", marker='o')
        plt.plot(val_acc, label="Validation Accuracy", marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("📈 Real-time Model Performance")
        plt.legend()
        plt.grid(True)
        plt.pause(0.1)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[MetricsCallback()]
)

# -----------------------------
# 🔟 Train & Save
# -----------------------------
plt.ion()
plt.figure()
trainer.train()
plt.ioff()
plt.show()

model.save_pretrained("./bert_toxic_model_multilabel_final")
tokenizer.save_pretrained("./bert_toxic_model_multilabel_final")

print("✅ Multi-label BERT model trained and saved successfully!")

