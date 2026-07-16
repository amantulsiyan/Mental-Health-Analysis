import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)

from config import DATA_DIR


def extract_features(
    csv_path=os.path.join(DATA_DIR, "processed", "train_clean.csv"),
    output_dir=os.path.join(DATA_DIR, "features"),
    max_features=5000,
):
    """
    Generate Bag-of-Words and TF-IDF features
    from the cleaned training dataset.
    """

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Loading cleaned dataset...")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")

    texts = df["text"].fillna("").astype(str)

    print(f"Documents: {len(texts)}")

    print("\nCreating Bag-of-Words vectorizer...")

    bow_vectorizer = CountVectorizer(
        max_features=max_features
    )

    bow_features = bow_vectorizer.fit_transform(texts)

    print("Creating TF-IDF vectorizer...")

    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features
    )

    tfidf_features = tfidf_vectorizer.fit_transform(texts)

    # Save vectorizers
    joblib.dump(
        bow_vectorizer,
        os.path.join(output_dir, "bow_vectorizer.pkl"),
    )

    joblib.dump(
        tfidf_vectorizer,
        os.path.join(output_dir, "tfidf_vectorizer.pkl"),
    )

    # Save feature matrices
    joblib.dump(
        bow_features,
        os.path.join(output_dir, "bow_features.pkl"),
    )

    joblib.dump(
        tfidf_features,
        os.path.join(output_dir, "tfidf_features.pkl"),
    )

    print("\nFeature Extraction Summary")
    print("-" * 40)

    print(f"BoW Shape      : {bow_features.shape}")
    print(f"TF-IDF Shape   : {tfidf_features.shape}")

    print(
        f"BoW Vocabulary : {len(bow_vectorizer.vocabulary_)}"
    )

    print(
        f"TF-IDF Vocabulary : {len(tfidf_vectorizer.vocabulary_)}"
    )

    print("\nFiles Saved")

    print(
        os.path.join(output_dir, "bow_vectorizer.pkl")
    )

    print(
        os.path.join(output_dir, "tfidf_vectorizer.pkl")
    )

    print(
        os.path.join(output_dir, "bow_features.pkl")
    )

    print(
        os.path.join(output_dir, "tfidf_features.pkl")
    )

    print("\nFeature extraction completed successfully!")


if __name__ == "__main__":
    extract_features()