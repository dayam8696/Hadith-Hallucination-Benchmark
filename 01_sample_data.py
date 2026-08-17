"""
Sample exactly 1000 random hadiths per book (stratified, reproducible).

Reads dataset/all_hadiths_clean.csv and writes data/sampled_1000_per_book.csv
"""

import pandas as pd

RANDOM_SEED = 42
SAMPLES_PER_BOOK = 1000

INPUT_PATH = "dataset/all_hadiths_clean.csv"
OUTPUT_PATH = "data/sampled_1000_per_book.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Strip whitespace from string columns that matter for grouping/joining
    df["source"] = df["source"].astype(str).str.strip()

    parts = [
        group.sample(n=min(SAMPLES_PER_BOOK, len(group)), random_state=RANDOM_SEED)
        for _, group in df.groupby("source")
    ]
    sampled = pd.concat(parts, ignore_index=True)

    sampled.to_csv(OUTPUT_PATH, index=False)

    print("Sampling complete.")
    print(f"Saved to: {OUTPUT_PATH}")
    print()
    print("Counts per book:")
    print(sampled["source"].value_counts())
    print()
    print(f"Total sampled hadiths: {len(sampled)}")


if __name__ == "__main__":
    main()
