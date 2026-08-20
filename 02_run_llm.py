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


REASONING_MODEL_HINTS = ["deepseek", "r1", "qwq", "o1", "qwen3"]


def call_ollama(model: str, prompt: str) -> str:
    # Reasoning models (e.g. deepseek-r1) spend part of num_predict on a
    # hidden "thinking" trace before writing the visible answer -- a tight
    # token cap can run out mid-thought, cutting the real answer short. Give
    # those models a much larger budget; other models keep a tight cap to
    # prevent runaway repetition.
    is_reasoning_model = any(hint in model.lower() for hint in REASONING_MODEL_HINTS)
    num_predict = 4000 if is_reasoning_model else 300

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": num_predict,
                "repeat_penalty": 1.3,
                "temperature": 0.2,
            },
        },
        timeout=420,
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
