"""
vader_module.py

Provides reusable VADER sentiment analysis utilities for the
Mental Health Analysis project.

Functions
---------
analyze_sentiment(text)
    Analyze sentiment of a single Reddit post.

generate_corpus_statistics()
    Generate average sentiment statistics for each class
    and save a visualization.
"""


import os

import matplotlib.pyplot as plt
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config import (
    DATA_DIR,
    CHARTS_DIR,
    CLASSES
)
from preprocess import basic_clean

analyzer = SentimentIntensityAnalyzer()


def analyze_sentiment(text):
    text = basic_clean(text)

    scores = analyzer.polarity_scores(text)

    compound = scores["compound"]

    if compound > 0.05:
        label = "Positive"
    elif compound < -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "compound": compound,
        "positive": scores["pos"],
        "negative": scores["neg"],
        "neutral": scores["neu"],
        "label": label
    }

def generate_corpus_statistics():
    train_path = os.path.join(DATA_DIR, "train.csv")
    df = pd.read_csv(train_path)

    # Calculate VADER compound score for every post
    df["compound"] = df["text"].apply(
        lambda text: analyze_sentiment(text)["compound"]
    )

    # Average compound score per class
    sentiment_stats = (
        df.groupby("label")["compound"]
        .mean()
        .reindex(CLASSES)
    )

    print("\nAverage Compound Sentiment\n")
    print(sentiment_stats)

    # Create charts directory if it doesn't exist
    os.makedirs(CHARTS_DIR, exist_ok=True)

    # Colors for each class
    colors = [
        "#4C72B0",  # depression
        "#DD8452",  # anxiety
        "#8172B2",  # bipolar
        "#C44E52",  # ptsd
        "#55A868",  # normal
    ]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        sentiment_stats.index,
        sentiment_stats.values,
        color=colors
    )

    plt.title(
        "Average VADER Compound Sentiment by Mental Health Category",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Mental Health Category", fontsize=12)
    plt.ylabel("Average Compound Score", fontsize=12)

    plt.axhline(0, color="black", linewidth=1)

    # Display values above bars
# Display values above bars
    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            y = height + 0.02
            va = "bottom"
        else:
            y = height - 0.03
            va = "top"

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{height:.2f}",
            ha="center",
            va=va,
            fontsize=10
        )

    plt.tight_layout()

    chart_path = os.path.join(CHARTS_DIR, "sentiment_by_class.png")

    plt.savefig(chart_path, dpi=150)
    plt.close()

    print(f"\nSaved chart to:\n{chart_path}")

    return sentiment_stats

if __name__ == "__main__":
    sample = "I feel hopeless and exhausted. Nothing is getting better."

    result = analyze_sentiment(sample)
    print(result)

    print("\nGenerating corpus statistics...\n")
    generate_corpus_statistics()