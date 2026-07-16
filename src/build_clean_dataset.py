import os
import pandas as pd

from preprocess import preprocess_dataframe
from config import DATA_DIR


def build_dataset(split):
    input_path = os.path.join(DATA_DIR, f"{split}.csv")
    output_dir = os.path.join(DATA_DIR, "processed")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading {split}.csv...")
    df = pd.read_csv(input_path)

    print(f"Original samples: {len(df)}")

    # RoBERTa preprocessing
    clean_df = preprocess_dataframe(df, branch="roberta")

    output_path = os.path.join(output_dir, f"{split}_clean.csv")
    clean_df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")
    print(f"Remaining samples: {len(clean_df)}")

    removed = len(df) - len(clean_df)
    print(f"Removed samples: {removed}")

    return clean_df


if __name__ == "__main__":
    print("=" * 50)
    print("Building Clean Datasets")
    print("=" * 50)

    train_df = build_dataset("train")
    val_df = build_dataset("val")
    test_df = build_dataset("test")

    print("\nDone!")