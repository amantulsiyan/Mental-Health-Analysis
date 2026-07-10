import os
import pandas as pd
from wordcloud import WordCloud, STOPWORDS


def generate_wordclouds(
    csv_path="data/train.csv",
    output_dir="assets/wordclouds"
):
    """
    Generate a word cloud for each mental health label
    and save it inside assets/wordclouds.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    df = pd.read_csv(csv_path)

    # Check required columns
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "Dataset must contain 'text' and 'label' columns."
        )

    # Generate word cloud for each label
    for label in sorted(df["label"].unique()):

        print(f"Generating word cloud for: {label}")

        # Combine all text belonging to the current label
        text = " ".join(
            df[df["label"] == label]["text"]
            .dropna()
            .astype(str)
        )

        # Create Word Cloud
        wordcloud = WordCloud(
            width=1200,
            height=700,
            background_color="white",
            stopwords=STOPWORDS,
            collocations=False,
            max_words=250,
            random_state=42
        ).generate(text)

        # Safe filename
        filename = f"{label.lower().replace(' ', '_')}.png"

        filepath = os.path.join(output_dir, filename)

        # Save image
        wordcloud.to_file(filepath)

        print(f"Saved: {filepath}")

    print("\nAll word clouds generated successfully!")


if __name__ == "__main__":
    generate_wordclouds()