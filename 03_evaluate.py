"""
Evaluate an LLM results CSV against ground truth using BLEU (text match)
and multilingual sentence-embedding cosine similarity (theme match).

Usage:
    python 03_evaluate.py --file results/mistral_english.csv
"""

import argparse
import os

import nltk
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BLEU_THRESHOLD = 0.4
COSINE_HIGH_THRESHOLD = 0.8
COSINE_LOW_THRESHOLD = 0.5


def ensure_nltk_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab")
        except Exception:
            pass


def compute_bleu(reference: str, hypothesis: str, smoothing) -> float:
    if not isinstance(reference, str) or not isinstance(hypothesis, str):
        return 0.0
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)


def classify(bleu: float, cosine: float) -> str:
    if cosine < COSINE_LOW_THRESHOLD:
        return "hallucinated"
    if bleu > BLEU_THRESHOLD and cosine > COSINE_HIGH_THRESHOLD:
        return "correct"
    if bleu < BLEU_THRESHOLD and cosine > COSINE_HIGH_THRESHOLD:
        return "theme_only"
    return "partial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to results CSV, e.g. results/mistral_english.csv")
    args = parser.parse_args()

    ensure_nltk_punkt()

    df = pd.read_csv(args.file)
    lang = df["lang"].iloc[0] if "lang" in df.columns and len(df) else "english"
    ground_truth_col = "ground_truth_en" if lang == "english" else "ground_truth_ar"

    smoothing = SmoothingFunction().method1
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    bleu_scores = []
    cosine_scores = []
    labels = []

    responses = df["llm_response"].fillna("").astype(str).tolist()
    references = df[ground_truth_col].fillna("").astype(str).tolist()

    print("Computing embeddings...")
    response_embeddings = embedder.encode(responses, convert_to_tensor=True, show_progress_bar=True)
    reference_embeddings = embedder.encode(references, convert_to_tensor=True, show_progress_bar=True)

    print("Computing BLEU and cosine scores...")
    for i in tqdm(range(len(df))):
        reference = references[i]
        hypothesis = responses[i]

        if hypothesis.strip().upper() == "ERROR" or not hypothesis.strip():
            bleu = 0.0
            cosine = 0.0
        else:
            bleu = compute_bleu(reference, hypothesis, smoothing)
            cosine = float(util.cos_sim(response_embeddings[i], reference_embeddings[i]).item())
            cosine = max(0.0, min(1.0, cosine))

        bleu_scores.append(bleu)
        cosine_scores.append(cosine)
        labels.append(classify(bleu, cosine))

    df["bleu_score"] = bleu_scores
    df["cosine_score"] = cosine_scores
    df["label"] = labels

    base_name = os.path.splitext(os.path.basename(args.file))[0]
    output_path = f"results/{base_name}_evaluated.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved evaluated results to: {output_path}")

    print("\n=== Label counts ===")
    print(df["label"].value_counts())

    print("\n=== Average scores per book ===")
    summary = df.groupby("source")[["bleu_score", "cosine_score"]].mean()
    print(summary)


if __name__ == "__main__":
    main()
