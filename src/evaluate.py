import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

print("[1/10] Downloading NLTK stopwords...")
nltk.download('stopwords')

from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
import json

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.preprocessing import label_binarize

from config import *
from preprocess import preprocess_dataframe
from train import MentalHealthDataset


# ── Toggle Inference ─────────────────────────────────────────

# True  → run model inference again
# False → load saved .npy outputs instantly
RUN_INFERENCE = False


# ── Device Setup ─────────────────────────────────────────────

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print(f"[2/10] Device selected: {device}")


# ── Load Model ───────────────────────────────────────────────

print("[3/10] Loading trained RoBERTa model...")

model = AutoModelForSequenceClassification.from_pretrained(
    os.path.join(MODEL_DIR, 'best_model')
)

model.to(device)
model.eval()

print("[4/10] Model loaded and moved to device successfully.")


# ── Load Tokenizer ───────────────────────────────────────────

print("[5/10] Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(MODEL_DIR, 'best_model')
)

print("[6/10] Tokenizer loaded successfully.")


# ── Load Dataset ─────────────────────────────────────────────

print("[7/10] Loading test dataset...")

test_df = pd.read_csv(
    os.path.join(DATA_DIR, "test.csv")
)

print(f"     Raw test samples: {len(test_df)}")


# ── Preprocessing ────────────────────────────────────────────

print("[8/10] Preprocessing test dataset using RoBERTa branch...")

test_df = preprocess_dataframe(
    test_df,
    branch='roberta'
)

print(f"     Samples after preprocessing/filtering: {len(test_df)}")


# ── Dataset + DataLoader ─────────────────────────────────────

print("[9/10] Creating dataset and dataloader...")

test_dataset = MentalHealthDataset(test_df, tokenizer)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"     Total batches: {len(test_loader)}")


# ── Run Inference OR Load Saved Outputs ──────────────────────

if RUN_INFERENCE:

    print("[10/10] Starting model inference on test set...")

    all_preds = []
    all_labels = []
    all_probs = []
    all_confidences = []

    with torch.no_grad():

        for batch_idx, batch in enumerate(test_loader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            logits = outputs.logits

            probs = torch.softmax(logits, dim=1)

            preds = torch.argmax(probs, dim=1)

            confidences = torch.max(probs, dim=1).values

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

            # Progress update every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(
                    f"     Processed batch {batch_idx + 1}/{len(test_loader)}"
                )

    # ── Convert to NumPy Arrays ──────────────────────────────

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_confidences = np.array(all_confidences)

    print("\nInference completed successfully.")
    print(f"Total samples evaluated: {len(all_labels)}")
    print(f"Predictions shape: {all_preds.shape}")
    print(f"Probabilities shape: {all_probs.shape}")
    print(f"Confidence scores shape: {all_confidences.shape}")

    # ── Save Inference Outputs ───────────────────────────────

    print("\nSaving inference outputs...")

    os.makedirs(LOGS_DIR, exist_ok=True)

    np.save(
        os.path.join(LOGS_DIR, "all_preds.npy"),
        all_preds
    )

    np.save(
        os.path.join(LOGS_DIR, "all_labels.npy"),
        all_labels
    )

    np.save(
        os.path.join(LOGS_DIR, "all_probs.npy"),
        all_probs
    )

    np.save(
        os.path.join(LOGS_DIR, "all_confidences.npy"),
        all_confidences
    )

    print("Inference outputs saved successfully.")
    print(f"Saved files to: {LOGS_DIR}")


else:

    print("\nLoading previously saved inference outputs...")

    all_preds = np.load(
        os.path.join(LOGS_DIR, "all_preds.npy")
    )

    all_labels = np.load(
        os.path.join(LOGS_DIR, "all_labels.npy")
    )

    all_probs = np.load(
        os.path.join(LOGS_DIR, "all_probs.npy")
    )

    all_confidences = np.load(
        os.path.join(LOGS_DIR, "all_confidences.npy")
    )

    print("Saved inference outputs loaded successfully.")

    print(f"Predictions shape: {all_preds.shape}")
    print(f"Labels shape: {all_labels.shape}")
    print(f"Probabilities shape: {all_probs.shape}")
    print(f"Confidence scores shape: {all_confidences.shape}")


# ── Classification Report ────────────────────────────────────

print("\nGenerating classification report...")

report = classification_report(
    all_labels,
    all_preds,
    target_names=CLASSES,
    digits=4
)

print("\n===== Classification Report =====\n")
print(report)
# ── Save Classification Report ──────────────────────────────

classification_report_path = os.path.join(
    LOGS_DIR,
    "classification_report.txt"
)

with open(classification_report_path, "w") as f:
    f.write(report)

print(
    f"\nClassification report saved to:\n"
    f"{classification_report_path}"
)


# ── Confusion Matrix ─────────────────────────────────────────

print("\nGenerating confusion matrix...")

os.makedirs(CHARTS_DIR, exist_ok=True)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASSES,
    yticklabels=CLASSES
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("RoBERTa Confusion Matrix")

conf_matrix_path = os.path.join(
    CHARTS_DIR,
    "roberta_confusion_matrix.png"
)

plt.savefig(conf_matrix_path, bbox_inches='tight')
plt.close()

print(f"Confusion matrix saved to:\n{conf_matrix_path}")

# ── ROC Curve ────────────────────────────────────────────────

print("\nGenerating ROC curves...")

y_true_bin = label_binarize(
    all_labels,
    classes=[0, 1, 2, 3, 4]
)

plt.figure(figsize=(8, 6))

for i in range(NUM_CLASSES):

    fpr, tpr, _ = roc_curve(
        y_true_bin[:, i],
        all_probs[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{CLASSES[i]} (AUC = {roc_auc:.4f})"
    )


# Random classifier diagonal
plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--'
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curves")

plt.legend(loc="lower right")

roc_curve_path = os.path.join(
    CHARTS_DIR,
    "roc_curves.png"
)

plt.savefig(roc_curve_path, bbox_inches='tight')
plt.close()

print(f"ROC curves saved to:\n{roc_curve_path}")

#Error Analysis

# ── Error Analysis CSV ───────────────────────────────────────

print("\nGenerating error analysis CSV...")

error_rows = []

for i in range(len(all_labels)):

    true_label = CLASSES[all_labels[i]]
    predicted_label = CLASSES[all_preds[i]]

    # Only store incorrect predictions
    if all_labels[i] != all_preds[i]:

        error_rows.append({

            "text": test_df.iloc[i]['text'],

            "true_label": true_label,

            "predicted_label": predicted_label,

            "confidence": float(all_confidences[i])

        })


# Convert to DataFrame
error_df = pd.DataFrame(error_rows)

# Save CSV
error_csv_path = os.path.join(
    ASSETS_DIR,
    "error_analysis.csv"
)

error_df.to_csv(
    error_csv_path,
    index=False
)

print(f"Total misclassified samples: {len(error_df)}")
print(f"Error analysis CSV saved to:\n{error_csv_path}")

# ── Metrics JSON Saving ──────────────────────────────────────

print("\nCalculating final evaluation metrics...")

accuracy = accuracy_score(
    all_labels,
    all_preds
)

macro_precision = precision_score(
    all_labels,
    all_preds,
    average=AVERAGING
)

macro_recall = recall_score(
    all_labels,
    all_preds,
    average=AVERAGING
)

macro_f1 = f1_score(
    all_labels,
    all_preds,
    average=AVERAGING
)

metrics = {

    "accuracy": round(float(accuracy), 4),

    "macro_precision": round(
        float(macro_precision), 4
    ),

    "macro_recall": round(
        float(macro_recall), 4
    ),

    "macro_f1": round(
        float(macro_f1), 4
    ),

    "num_samples": int(len(all_labels)),

    "model_name": MODEL_NAME,

    "batch_size": BATCH_SIZE,

    "max_len": MAX_LEN,

    "num_classes": NUM_CLASSES
}


metrics_json_path = os.path.join(
    LOGS_DIR,
    "eval_results.json"
)

with open(metrics_json_path, "w") as f:

    json.dump(
        metrics,
        f,
        indent=4
    )

print("Evaluation metrics saved successfully.")
print(f"Metrics JSON saved to:\n{metrics_json_path}")


print("\n===== FINAL METRICS =====")

for key, value in metrics.items():
    print(f"{key}: {value}")