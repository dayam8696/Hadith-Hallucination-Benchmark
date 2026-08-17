"""
Combine all evaluated results CSVs in results/ into a single comparison table.

Usage:
    python 04_results_summary.py
"""

import glob
import os

import pandas as pd

RESULTS_DIR = "results"
OUTPUT_PATH = "results/final_summary.csv"


def main():
    files = sorted(glob.glob(f"{RESULTS_DIR}/*_evaluated.csv"))
    if not files:
        print(f"No *_evaluated.csv files found in {RESULTS_DIR}/")
        return

    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    rows = []
    for (model, lang), group in combined.groupby(["model", "lang"]):
        total = len(group)
        label_counts = group["label"].value_counts(normalize=True) * 100

        rows.append({
            "model": model,
            "lang": lang,
            "avg_bleu": group["bleu_score"].mean(),
            "avg_cosine": group["cosine_score"].mean(),
            "pct_correct": label_counts.get("correct", 0.0),
            "pct_theme_only": label_counts.get("theme_only", 0.0),
            "pct_hallucinated": label_counts.get("hallucinated", 0.0),
            "pct_partial": label_counts.get("partial", 0.0),
            "n": total,
        })

    summary = pd.DataFrame(rows).sort_values(["model", "lang"]).reset_index(drop=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print(f"Combined {len(files)} evaluated file(s) from {RESULTS_DIR}/")
    print(f"Saved final summary to: {OUTPUT_PATH}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
