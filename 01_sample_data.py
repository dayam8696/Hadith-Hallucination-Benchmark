"""
Sample exactly 500 random hadiths per book (stratified, reproducible).

Reads dataset/all_hadiths_clean.csv and writes one CSV per book into data/,
e.g. data/sampled_500_Sahih_Bukhari.csv, each with 500 rows from that book.
"""

import re

import pandas as pd

RANDOM_SEED = 42
SAMPLES_PER_BOOK = 500

INPUT_PATH = "dataset/all_hadiths_clean.csv"
OUTPUT_DIR = "data"


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def main():
    df = pd.read_csv(INPUT_PATH)

    # Strip whitespace from string columns that matter for grouping/joining
    df["source"] = df["source"].astype(str).str.strip()

    print("Sampling complete.")
    print()
    total = 0
    for source, group in df.groupby("source"):
        sampled = group.sample(n=min(SAMPLES_PER_BOOK, len(group)), random_state=RANDOM_SEED)
        output_path = f"{OUTPUT_DIR}/sampled_500_{slugify(source)}.csv"
        sampled.to_csv(output_path, index=False)
        print(f"{source}: {len(sampled)} rows -> {output_path}")
        total += len(sampled)

    print()
    print(f"Total sampled hadiths across all books: {total}")


if __name__ == "__main__":
    main()
