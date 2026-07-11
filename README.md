# 🧠 Mental Health Monitoring Through Social Media Analytics

An end-to-end Natural Language Processing (NLP) system for detecting mental health conditions from social media posts using **RoBERTa**, **VADER Sentiment Analysis**, **LIME Explainability**, and an interactive **Streamlit Dashboard**.

---

# 🚀 Features

- 🔍 Fine-tuned **RoBERTa** for 5-class mental health classification
- 😊 Parallel **VADER** sentiment analysis
- 📊 Exploratory Data Analysis (EDA)
- ☁️ Per-class **Word Clouds**
- 📈 Top-20 Frequent Word Analysis
- 🔤 Text Preprocessing Pipeline
- 📚 TF-IDF & Bag-of-Words Feature Engineering
- 🧠 LIME Explainability
- 📉 Confusion Matrix & ROC Curves
- 🌐 Interactive Streamlit Dashboard

---

# 📂 Dataset

### Source

Reddit Mental Health Corpus

The dataset contains posts collected from mental-health-related subreddits.

### Classes

- Depression
- Anxiety
- Bipolar
- PTSD
- Normal

### Dataset Statistics

| Split | Samples |
|--------|---------|
| Train | 19,529 |
| Validation | 2,441 |
| Test | 2,442 |
| **Total** | **24,412** |

The dataset is approximately balanced across all five classes.

---

# 🧹 Text Preprocessing

Two preprocessing pipelines are implemented.

## 1. RoBERTa Pipeline

Designed for transformer models.

Operations:

- URL removal
- HTML tag removal
- HTML entity removal
- Unicode normalization
- Whitespace normalization

This pipeline preserves sentence structure for transformer-based learning.

---

## 2. VADER / Traditional ML Pipeline

Designed for statistical analysis and feature engineering.

Operations:

- Lowercasing
- URL removal
- HTML removal
- Punctuation removal
- Stopword removal
- Custom stopword filtering
- Lemmatization

Used for:

- Frequent word analysis
- TF-IDF
- Bag-of-Words

---

# 📊 Exploratory Data Analysis

The project includes several exploratory visualizations:

- Class distribution
- Text length distribution
- Word Clouds
- Top-20 Frequent Words
- Confusion Matrix
- ROC Curves
- Error Analysis

---

# ☁️ Word Cloud Generation

Word clouds are automatically generated for every class.

Generated files:

```
assets/wordclouds/

anxiety.png
bipolar.png
depression.png
normal.png
ptsd.png
```

---

# 📈 Frequent Word Analysis

The project generates Top-20 frequent word charts for every class after VADER preprocessing.

Generated files:

```
assets/frequent_words/

anxiety_top20.png
bipolar_top20.png
depression_top20.png
normal_top20.png
ptsd_top20.png
```

---

# 📚 Feature Engineering

Traditional Machine Learning features are generated using:

- CountVectorizer (Bag-of-Words)
- TF-IDF Vectorizer

Generated artifacts:

```
data/features/

bow_vectorizer.pkl
tfidf_vectorizer.pkl
bow_features.pkl
tfidf_features.pkl
```

These can be reused without recomputing features.

---

# 🏗️ Project Structure

```text
Mental-Health-Analysis/
│
├── assets/
│   ├── charts/
│   ├── frequent_words/
│   ├── lime_examples/
│   ├── lime_html/
│   ├── wordclouds/
│   └── error_analysis.csv
│
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   ├── processed/
│   │   ├── train_clean.csv
│   │   ├── val_clean.csv
│   │   └── test_clean.csv
│   └── features/
│       ├── bow_features.pkl
│       ├── bow_vectorizer.pkl
│       ├── tfidf_features.pkl
│       └── tfidf_vectorizer.pkl
│
├── logs/
│
├── src/
│   ├── app.py
│   ├── baseline.py
│   ├── build_clean_dataset.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── feature_extraction.py
│   ├── frequent_words.py
│   ├── lime_explainer.py
│   ├── preprocess.py
│   ├── train.py
│   ├── vader_module.py
│   └── wordcloud_gen.py
│
├── README.md
└── requirements.txt
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone <repository-url>
```

Move into the project directory

```bash
cd Mental-Health-Analysis
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1 — Build Clean Dataset

```bash
python src/build_clean_dataset.py
```

---

## Step 2 — Generate Word Clouds

```bash
python src/wordcloud_gen.py
```

---

## Step 3 — Generate Frequent Word Charts

```bash
python src/frequent_words.py
```

---

## Step 4 — Generate TF-IDF & Bag-of-Words Features

```bash
python src/feature_extraction.py
```

---

## Step 5 — Train RoBERTa

```bash
python src/train.py
```

---

## Step 6 — Evaluate the Model

```bash
python src/evaluate.py
```

Outputs:

- Classification Report
- Confusion Matrix
- ROC Curves
- Error Analysis

---

## Step 7 — Generate LIME Explanations

```bash
python src/lime_explainer.py
```

---

## Step 8 — Launch Streamlit Dashboard

```bash
streamlit run src/app.py
```

---

# 📈 Results

## RoBERTa Performance

| Class | Precision | Recall | F1 |
|--------|-----------|--------|------|
| Depression | 0.7985 | 0.8400 | 0.8187 |
| Anxiety | 0.8489 | 0.8540 | 0.8514 |
| Bipolar | 0.8995 | 0.8710 | 0.8851 |
| PTSD | 0.9204 | 0.9020 | 0.9111 |
| Normal | 0.9534 | 0.9439 | 0.9486 |

### Overall Performance

| Metric | Score |
|---------|--------|
| Accuracy | **88.24%** |
| Macro Precision | **0.8842** |
| Macro Recall | **0.8822** |
| Macro F1 | **0.8830** |

---

## Baseline Model

TF-IDF + Logistic Regression

| Split | Accuracy | Macro F1 |
|---------|---------|----------|
| Validation | 83.20% | 0.8309 |
| Test | 84.11% | 0.8412 |

---

## Performance Improvement

| Metric | Baseline | RoBERTa |
|---------|----------|----------|
| Accuracy | 84.11% | **88.24%** |
| Macro F1 | 0.8412 | **0.8830** |

---

# 🔍 Explainability

The project uses **LIME** to explain model predictions.

For every explanation it generates:

- Prediction
- Confidence
- Top influential words
- Probability distribution
- HTML explanation
- PNG visualization

Outputs:

```
assets/lime_html/
assets/lime_examples/
```

---

# ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Base Model | roberta-base |
| Max Length | 512 |
| Batch Size | 8 |
| Effective Batch Size | 32 |
| Learning Rate | 2e-5 |
| Weight Decay | 0.01 |
| Epochs | 5 |
| Warmup Ratio | 0.1 |
| Optimizer | AdamW |
| Early Stopping | Patience = 2 |
| Random Seed | 42 |

---

# 👥 Team

| Member | Contribution |
|----------|--------------|
| **Adithya KL** | Dataset preparation, EDA, RoBERTa training, baseline model |
| **Aman Tulsiyan** | Evaluation pipeline, LIME explainability, error analysis |
| **Apoorv Anand** | Text preprocessing, VADER sentiment analysis, word cloud generation, frequent word analysis, TF-IDF & Bag-of-Words feature engineering, Streamlit dashboard |

---

# 🤗 Model Checkpoint

The trained model is available on Hugging Face.

**Repository**

https://huggingface.co/Adithya-257/mental-health-roberta

---

# 📜 License

This project was developed for academic purposes as part of a Machine Learning course project.