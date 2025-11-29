import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

label_cols = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']

preds = np.load("preds.npy")
test_labels = np.load("labels.npy")

cm = multilabel_confusion_matrix(test_labels, preds)

for i, label in enumerate(label_cols):
    plt.figure()
    sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {label}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
