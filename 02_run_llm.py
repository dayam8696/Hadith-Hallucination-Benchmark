"""
Run a local Ollama model over the sampled hadiths and record its responses.

Usage:
    python 02_run_llm.py --model mistral --lang english
    python 02_run_llm.py --model llama3 --lang arabic
"""

import argparse
import time

import pandas as pd
import requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
DATA_PATH = "data/sampled_1000_per_book.csv"
PROMPT_DIR = "prompts"
RESULTS_DIR = "results"
DELAY_SECONDS = 0.5


def load_prompt_template(lang: str) -> str:
    filename = "prompt_english.txt" if lang == "english" else "prompt_arabic.txt"
    with open(f"{PROMPT_DIR}/{filename}", "r", encoding="utf-8") as f:
        return f.read()


def call_ollama(model: str, prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Ollama model name, e.g. mistral, llama3, deepseek-r1")
    parser.add_argument("--lang", required=True, choices=["english", "arabic"], help="Prompt language")
    parser.add_argument("--data", default=DATA_PATH, help="Path to sampled hadiths CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    template = load_prompt_template(args.lang)

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{args.model} ({args.lang})"):
        prompt = template.format(
            source=row["source"],
            chapter_no=row["chapter_no"],
            chapter=row["chapter"],
            hadith_no=row["hadith_no"],
        )

        try:
            llm_response = call_ollama(args.model, prompt)
        except Exception as e:
            llm_response = "ERROR"
            tqdm.write(f"Ollama call failed for hadith_id={row['hadith_id']}: {e}")

        rows.append({
            "hadith_id": row["hadith_id"],
            "source": row["source"],
            "chapter_no": row["chapter_no"],
            "hadith_no": row["hadith_no"],
            "chapter": row["chapter"],
            "ground_truth_en": row["text_en"],
            "ground_truth_ar": row["text_ar"],
            "llm_response": llm_response,
            "model": args.model,
            "lang": args.lang,
        })

        time.sleep(DELAY_SECONDS)

    out_df = pd.DataFrame(rows)
    safe_model_name = args.model.replace(":", "-").replace("/", "-")
    output_path = f"{RESULTS_DIR}/{safe_model_name}_{args.lang}.csv"
    out_df.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")

# main
if __name__ == "__main__":
    main()
