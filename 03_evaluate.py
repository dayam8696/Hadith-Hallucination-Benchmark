"""
Evaluate an LLM results CSV against ground truth.

Text match (word match) is computed for every row using BLEU (nltk
sentence_bleu with smoothing) -- strict n-gram overlap, so it naturally
scores near zero for unrelated/fabricated content in both languages.

Theme match (semantic/meaning match) is computed for every row using
multilingual sentence-embedding cosine similarity
(sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), for both
English and Arabic.

Rows where the model refused to answer or asked for clarification instead
of producing a hadith (e.g. "I don't have access to...", "Please provide
the reference", "لا أستطيع مساعدتك") are detected before scoring and
labeled "refused" directly, without being scored against ground truth.

Usage:
    python 03_evaluate.py --file results/mistral-7b_english.csv
    python 03_evaluate.py --file results/mistral-7b_arabic.csv
"""

import argparse
import os

import nltk
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sentence_transformers import SentenceTransformer, util

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TEXT_MATCH_THRESHOLD = 0.4
THEME_HIGH_THRESHOLD = 0.8
THEME_LOW_THRESHOLD = 0.5

REFUSAL_PHRASES_EN = [
    "i do not have direct access",
    "i don't have direct access",
    "i do not have access to",
    "i don't have access to",
    "i am unable to verify",
    "i'm unable to verify",
    "please provide the reference",
    "please go ahead and provide",
    "i'm ready to provide",
    "i am ready to provide",
    "what is the reference",
    "i cannot help you with that",
    "i can't help you with that",
    "large language model, i don't have",
    "large language model, i do not have",
    "i will wait for your confirmation",
    "please provide me with the",
    "could you provide the reference",
]

REFUSAL_PHRASES_AR = [
    "لا أستطيع مساعدتك",
    "لا يمكنني مساعدتك",
    "لا يوجد محتوى",
    "لا أملك",
    "ليس لدي",
    "غير قادر على",
    "من فضلك زودني",
    "الرجاء تزويدي",
    "يرجى تزويدي",
    "لا يمكنني تقديم",
]

# A response that hedges ("I'm ready to provide...") but then still goes on
# to fabricate a hadith is a hallucination, not a refusal -- only short
# responses dominated by the refusal phrase itself count as a true refusal.
REFUSAL_MAX_LENGTH = 150


def ensure_nltk_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt")
        except Exception:
            pass


def is_refusal(response: str, lang: str) -> bool:
    text = response.strip()
    if not text or len(text) > REFUSAL_MAX_LENGTH:
        return False
    phrases = REFUSAL_PHRASES_EN if lang == "english" else REFUSAL_PHRASES_AR
    return any(phrase.lower() in text.lower() for phrase in phrases)


def compute_bleu(reference: str, hypothesis: str, smoothing) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothing)


def classify(text_score: float, theme_score: float) -> str:
    if theme_score < THEME_LOW_THRESHOLD:
        return "hallucinated"
    if text_score > TEXT_MATCH_THRESHOLD and theme_score > THEME_HIGH_THRESHOLD:
        return "correct"
    if text_score < TEXT_MATCH_THRESHOLD and theme_score > THEME_HIGH_THRESHOLD:
        return "theme_only"
    return "partial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to results CSV, e.g. results/mistral-7b_english.csv")
    args = parser.parse_args()

    ensure_nltk_punkt()

    df = pd.read_csv(args.file)
    lang = df["lang"].iloc[0] if "lang" in df.columns and len(df) else "english"
    ground_truth_col = "ground_truth_en" if lang == "english" else "ground_truth_ar"

    responses = df["llm_response"].fillna("").astype(str).tolist()
    references = df[ground_truth_col].fillna("").astype(str).tolist()

    error_mask = [r.strip().upper() == "ERROR" or not r.strip() for r in responses]
    refusal_mask = [is_refusal(r, lang) for r in responses]
    skip_mask = [error_mask[i] or refusal_mask[i] for i in range(len(df))]

    n_refused = sum(refusal_mask)
    if n_refused:
        print(f"Detected {n_refused} refusal/non-answer response(s) -- these will be labeled 'refused' and excluded from scoring.")

    print("Computing text match (BLEU)...")
    smoothing = SmoothingFunction().method1
    text_match_scores = []
    for i in range(len(df)):
        if skip_mask[i]:
            text_match_scores.append(0.0)
        else:
            text_match_scores.append(compute_bleu(references[i], responses[i], smoothing))

    df["text_match_score"] = text_match_scores

    print(f"Computing theme match (cosine similarity, model={EMBEDDING_MODEL_NAME})...")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    safe_responses = [r if r.strip() else " " for r in responses]
    safe_references = [r if r.strip() else " " for r in references]
    response_embeddings = embedder.encode(safe_responses, convert_to_tensor=True, show_progress_bar=True)
    reference_embeddings = embedder.encode(safe_references, convert_to_tensor=True, show_progress_bar=True)

    theme_match_scores = []
    for i in range(len(df)):
        if skip_mask[i]:
            theme_match_scores.append(0.0)
            continue
        cosine = float(util.cos_sim(response_embeddings[i], reference_embeddings[i]).item())
        theme_match_scores.append(max(0.0, min(1.0, cosine)))

    df["theme_match_score"] = theme_match_scores
    labels = []
    for i in range(len(df)):
        if refusal_mask[i]:
            labels.append("refused")
        else:
            labels.append(classify(text_match_scores[i], theme_match_scores[i]))
    df["label"] = labels

    base_name = os.path.splitext(os.path.basename(args.file))[0]
    output_path = f"results/{base_name}_evaluated.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved evaluated results to: {output_path}")

    print("\n=== Label counts ===")
    print(df["label"].value_counts())

    print("\n=== Average scores per book ===")
    score_cols = ["text_match_score", "theme_match_score"]
    print(df.groupby("source")[score_cols].mean())


if __name__ == "__main__":
    main()
