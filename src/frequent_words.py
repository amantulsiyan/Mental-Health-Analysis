import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

from config import DATA_DIR, ASSETS_DIR
from preprocess import vader_clean


def generate_frequent_words(
    csv_path=os.path.join(DATA_DIR, "processed", "train_clean.csv"),
    output_dir=os.path.join(ASSETS_DIR, "frequent_words"),
    top_n=20
):
    """
    Generate Top-N frequent word charts for each mental health label.

    Uses VADER preprocessing (stopword removal + lemmatization)
    to produce meaningful word frequency visualizations.
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(csv_path)

    # Check required columns
    required_columns = {"text", "label"}

    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    # Clean text for visualization
    print("Applying VADER preprocessing...\n")

    df["clean_text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .apply(vader_clean)
    )

    # Generate one chart for each label
    for label in sorted(df["label"].unique()):

        print(f"Generating chart for: {label}")

        # Combine all text of current class
        text = " ".join(
            df[df["label"] == label]["clean_text"]
        )

        # Tokenize
        words = [
            word
            for word in text.split()
            if len(word) > 2
        ]

        # Count frequency
        counter = Counter(words)

        # Remove extremely rare words
        counter = Counter({
            word: count
            for word, count in counter.items()
            if count > 1
        })

        most_common = counter.most_common(top_n)

        if len(most_common) == 0:
            print(f"Skipping {label} (no words found)\n")
            continue

        labels = [item[0] for item in most_common]
        counts = [item[1] for item in most_common]

        # Plot
        plt.figure(figsize=(12, 7))

        plt.barh(
            labels[::-1],
            counts[::-1]
        )

        plt.title(
            f"Top {top_n} Frequent Words - {label}",
            fontsize=16,
            fontweight="bold"
        )

        plt.xlabel("Frequency", fontsize=12)
        plt.ylabel("Words", fontsize=12)

        plt.grid(
            axis="x",
            linestyle="--",
            alpha=0.4
        )

        plt.tight_layout()

        filename = (
            f"{label.lower().replace(' ', '_')}"
            f"_top{top_n}.png"
        )

        filepath = os.path.join(
            output_dir,
            filename
        )

        plt.savefig(
            filepath,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved: {filepath}\n")

    print("=" * 60)
    print("All frequent word charts generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    generate_frequent_words()