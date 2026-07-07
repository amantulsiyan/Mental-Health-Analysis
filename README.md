# Mental Health Monitoring Through Social Media Analytics

An end-to-end mental health text analysis system using fine-tuned RoBERTa,
VADER sentiment analysis, LIME explainability, and a Streamlit dashboard.

## System Overview

- **Classification**: Fine-tuned `roberta-base` for 5-class mental health prediction (depression, anxiety, bipolar, PTSD, normal)
- **Sentiment**: VADER parallel sentiment scoring
- **Explainability**: LIME token-level prediction explanations
- **Dashboard**: Streamlit interactive visualization

## Dataset

- **Source**: Reddit Mental Health Corpus (5 subreddits: depression, anxiety, bipolar, PTSD, normal/jokes/fitness/relationships)
- **Size**: Up to 5,000 samples per class, balanced — ~25,000 total before split
- **Split**: 80% train / 10% validation / 10% test (stratified)
- **Test set**: 2,441 samples

## Project Structure

```
├── data/               # train.csv, val.csv, test.csv (gitignored)
├── models/
│   └── best_model/     # Saved RoBERTa checkpoint (gitignored, shared via HuggingFace Hub)
├── assets/
│   ├── charts/         # Training curves, confusion matrix, ROC curves, EDA plots
│   ├── lime_html/      # Per-prediction LIME HTML explanations
│   ├── lime_examples/  # Per-prediction LIME bar chart PNGs
│   ├── wordclouds/     # Per-class word clouds
│   └── error_analysis.csv
├── logs/
│   ├── eval_results.json
│   ├── classification_report.txt
│   ├── baseline_results.json
│   ├── runs.json
│   └── all_preds/labels/probs/confidences.npy
├── src/
│   ├── config.py           # All constants, paths, hyperparameters
│   ├── preprocess.py       # Text cleaning (RoBERTa + VADER branches)
│   ├── data_loader.py      # Data loading, balancing, EDA, splitting
│   ├── train.py            # RoBERTa fine-tuning loop
│   ├── baseline.py         # TF-IDF + Logistic Regression baseline
│   ├── evaluate.py         # Model evaluation, metrics, error analysis
│   ├── lime_explainer.py   # LIME explainability
│   ├── vader_module.py     # VADER sentiment scoring
│   ├── wordcloud_gen.py    # Word cloud generation
│   └── app.py              # Streamlit dashboard
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the Pipeline

```bash
# 1. Load and prepare data
python src/data_loader.py

# 2. Train RoBERTa
python src/train.py

# 3. Evaluate on test set
python src/evaluate.py

# 4. Run LIME explainability on case studies
python src/lime_explainer.py

# 5. Launch Streamlit dashboard
streamlit run src/app.py
```

## Results

### RoBERTa (Fine-tuned `roberta-base`)

| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Depression | 0.7985    | 0.8400 | 0.8187   | 500     |
| Anxiety    | 0.8489    | 0.8540 | 0.8514   | 500     |
| Bipolar    | 0.8995    | 0.8710 | 0.8851   | 442     |
| PTSD       | 0.9204    | 0.9020 | 0.9111   | 500     |
| Normal     | 0.9534    | 0.9439 | 0.9486   | 499     |
| **Macro Avg** | **0.8842** | **0.8822** | **0.8830** | 2441 |

- **Test Accuracy**: 88.24%
- **Macro F1**: 0.8830
- **Macro Precision**: 0.8842
- **Macro Recall**: 0.8822

### Baseline (TF-IDF + Logistic Regression)

| Split      | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|------------|----------|----------|-----------------|--------------|
| Validation | 0.8320   | 0.8309   | 0.8345          | 0.8306       |
| Test       | 0.8411   | 0.8412   | 0.8464          | 0.8399       |

### RoBERTa vs Baseline

| Metric      | Baseline | RoBERTa | Improvement |
|-------------|----------|---------|-------------|
| Macro F1    | 0.8412   | 0.8830  | **+0.0418** |
| Accuracy    | 0.8411   | 0.8824  | **+0.0413** |

## Evaluation Pipeline (`evaluate.py`)

- Loads the saved `best_model` checkpoint
- Supports `RUN_INFERENCE = True/False` toggle — re-run inference or load cached `.npy` outputs
- Generates:
  - Classification report (per-class precision, recall, F1)
  - Confusion matrix heatmap → `assets/charts/roberta_confusion_matrix.png`
  - Multiclass ROC curves with per-class AUC → `assets/charts/roc_curves.png`
  - Error analysis CSV (misclassified samples with confidence scores) → `assets/error_analysis.csv`
  - Metrics JSON → `logs/eval_results.json`

## LIME Explainability (`lime_explainer.py`)

- Uses `LimeTextExplainer` with bag-of-words mode
- `predict_proba()` wraps the RoBERTa model for LIME compatibility
- `explain_text(text, true_label)` returns:
  - Predicted class and confidence
  - Top 10 most influential tokens with importance scores
  - Per-class probability distribution
  - Inference time
  - Correct/incorrect prediction flag (when true label is provided)
- Outputs saved per prediction:
  - HTML explanation → `assets/lime_html/<class>_<timestamp>.html`
  - Bar chart PNG → `assets/lime_examples/<class>_<timestamp>.png`
- Ships with 10 hand-crafted case studies (2 per class) for demo/testing

## Training Configuration

| Hyperparameter       | Value                  |
|----------------------|------------------------|
| Base model           | `roberta-base`         |
| Max sequence length  | 512                    |
| Batch size           | 8 (effective: 32 w/ grad accum) |
| Gradient accumulation| 4 steps                |
| Learning rate        | 2e-5                   |
| Weight decay         | 0.01                   |
| Warmup ratio         | 0.1                    |
| Max epochs           | 5                      |
| Early stopping       | Patience = 2           |
| Optimizer            | AdamW                  |
| Mixed precision      | AMP (CUDA)             |
| Seed                 | 42                     |

## Team

| Member          | Contributions                                              |
|-----------------|------------------------------------------------------------|
| Adithya KL      | Data pipeline, EDA, RoBERTa training, baseline model       |
| Aman Tulsiyan   | Evaluation, LIME explainability, error analysis |
| Apoorv Anand    | VADER sentiment, word clouds, Streamlit dashboard          |

## Model Checkpoint

Available on HuggingFace Hub: [Adithya-257/mental-health-roberta](https://huggingface.co/Adithya-257/mental-health-roberta)
