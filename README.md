# Hadith Hallucination Benchmark

A benchmark pipeline that measures how much local LLMs (run via [Ollama](https://ollama.com)) **hallucinate** when asked to recall specific Hadith texts from reference (Book, Chapter, Hadith Number) alone — instead of quoting the real text, do they confidently invent one?

Dataset: 6 major Hadith collections — **Sahih Bukhari, Sahih Muslim, Sunan Abi Da'ud, Jami' al-Tirmidhi, Sunan an-Nasa'i, Sunan Ibn Majah**.

Models tested so far: **mistral:7b**, **deepseek-r1:8b**, **llama3.1:8b** (all running locally via Ollama).

---

## 1. Why this exists

Hadith texts are precise, sourced religious records. If an LLM is asked "what does Hadith #442 of Sunan Abi Da'ud say?" and it doesn't actually know, a bad outcome is a model that **confidently fabricates a plausible-sounding hadith** rather than admitting it doesn't know. This benchmark quantifies that failure mode at scale, across models and across languages (English and Arabic).

## 2. Project structure

```
hadith_benchmark/
├── dataset/
│   └── all_hadiths_clean.csv        # raw source data (id, hadith_id, source, chapter_no,
│                                     #   hadith_no, chapter, chain_indx, text_ar, text_en)
├── data/
│   └── sampled_1000_per_book.csv    # stratified sample, 1000 hadiths/book, seed=42
├── prompts/
│   ├── prompt_english.txt           # English prompt template
│   └── prompt_arabic.txt            # Arabic prompt template
├── results/
│   ├── {model}_{lang}.csv           # raw LLM output per model/language
│   ├── {model}_{lang}_evaluated.csv # + bleu_score, cosine_score, label
│   └── final_summary.csv            # combined comparison across all models
├── doc/
│   └── Hadith_Hallucination_Benchmark.pdf   # this project explained as a PDF
├── 01_sample_data.py
├── 02_run_llm.py
├── 03_evaluate.py
├── 04_results_summary.py
├── requirements.txt
└── README.md
```

## 3. How the pipeline works

### Step 1 — `01_sample_data.py`
Reads `dataset/all_hadiths_clean.csv`, strips whitespace from the `source` column, and draws a **stratified random sample of 1000 hadiths per book** (reproducible with `random_seed=42`). Saves to `data/sampled_1000_per_book.csv`. Prints a per-book count as a sanity check.

### Step 2 — `02_run_llm.py`
For a chosen `--model` (e.g. `mistral:7b`) and `--lang` (`english` or `arabic`):
1. Loads the matching prompt template from `prompts/`.
2. For each sampled hadith, fills in `{source}`, `{chapter_no}`, `{chapter}`, `{hadith_no}` — **the model is given only the reference, never the actual text** — and sends it to Ollama's local API (`http://localhost:11434`).
3. Records the model's raw response next to the real ground-truth text (`text_en` / `text_ar`).
4. Adds a progress bar (`tqdm`), a 0.5s delay between calls, and writes `"ERROR"` for any failed call instead of crashing.
5. Saves to `results/{model}_{lang}.csv` with columns:
   `hadith_id, source, chapter_no, hadith_no, chapter, ground_truth_en, ground_truth_ar, llm_response, model, lang`

**Prompt design note:** the prompt does *not* forbid the model from including the chain of narrators (isnad) — since real ground-truth text sometimes includes narrator attribution (e.g. "Narrated Abu Huraira:"), forcing the model to omit it would create an unfair, structurally different comparison.

### Step 3 — `03_evaluate.py`
Scores every row on **two independent axes**, because text-match and theme-match catch different failure modes:

- **Level 1 — BLEU score (text match)**: is the model quoting the *exact words* of the real hadith? Uses `nltk.translate.bleu_score.sentence_bleu` with smoothing, scored 0–1.
- **Level 2 — Cosine similarity (theme match)**: does the model's answer at least mean the same thing, even with different wording? Both texts are embedded with the multilingual sentence-embedding model **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**, and cosine similarity between the two vectors is computed, scored 0–1. This model runs **entirely locally** (downloaded once from Hugging Face, cached, no API calls after that) and works across English and Arabic with one shared model.

Each row is then classified:

| Label | Condition | Meaning |
|---|---|---|
| `correct` | BLEU > 0.4 **and** cosine > 0.8 | Model reproduced the real hadith closely |
| `theme_only` | BLEU < 0.4 **and** cosine > 0.8 | Right general idea, wrong/paraphrased wording |
| `hallucinated` | cosine < 0.5 | Model invented unrelated content |
| `partial` | everything else | Somewhere in between |

Output: `results/{file}_evaluated.csv` (all original columns + `bleu_score`, `cosine_score`, `label`), plus a printed summary of label counts and average scores per book.

### Step 4 — `04_results_summary.py`
Combines every `*_evaluated.csv` in `results/` into one table — average BLEU, average cosine, and the % breakdown of correct / theme_only / hallucinated / partial — grouped by model and language. Saves `results/final_summary.csv` and prints it.

## 4. Understanding BLEU vs. Cosine (in plain terms)

- **BLEU** = "did it use the *same words*?" It compares overlapping word sequences (n-grams) between the model's answer and the real hadith. Even a correct paraphrase scores near 0 if the wording differs — it is a strict, literal match.
- **Cosine similarity** = "did it mean the *same thing*?" Both texts are converted into numeric vectors by the embedding model, and the angle between those vectors is measured — a small angle (score near 1) means similar meaning, regardless of exact wording.

**Caveat found during testing:** for Arabic religious text, cosine similarity can be misleadingly high (0.7–0.96) even when the hadith content is completely fabricated, because formulaic phrases common to all hadiths (e.g. "صلى الله عليه وسلم", "قال رسول الله") push the embeddings close together regardless of actual content. This is why BLEU and cosine are used **together**, not in isolation — a hallucinated hadith can look thematically close while being factually invented.

## 5. Models evaluated (pilot run)

This is a small-scale pilot (1 hadith per book, 6 books = 6 hadiths, run in both English and Arabic) used to validate the pipeline end-to-end before scaling to the full 1000-per-book sample. Three local Ollama models were tested:

- **mistral:7b**
- **deepseek-r1:8b**
- **llama3.1:8b**

### Pilot results (`results/final_summary.csv`)

| Model | Lang | Avg BLEU | Avg Cosine | % Correct | % Theme Only | % Hallucinated | % Partial |
|---|---|---|---|---|---|---|---|
| deepseek-r1:8b | arabic | 0.011 | 0.652 | 0% | 33.3% | 33.3% | 33.3% |
| deepseek-r1:8b | english | 0.003 | 0.431 | 0% | 0% | 33.3% | 66.7% |
| llama3.1:8b | arabic | 0.015 | 0.478 | 0% | 0% | 50.0% | 50.0% |
| llama3.1:8b | english | 0.020 | 0.402 | 0% | 0% | 66.7% | 33.3% |
| mistral:7b | arabic | 0.008 | 0.725 | 0% | 33.3% | 16.7% | 50.0% |
| mistral:7b | english | 0.007 | 0.589 | 0% | 0% | 16.7% | 83.3% |

**Observations:**
- **0% "correct" across every model and language** in this pilot — no model reproduced exact hadith wording from a bare reference (book/chapter/hadith number alone).
- **llama3.1:8b hallucinated the most**, especially in English (66.7%).
- **mistral:7b hallucinated the least** and had the highest average cosine — though part of that is the formulaic-Arabic-phrasing effect described above, not necessarily better recall.
- **English hallucination rates were higher than Arabic** for deepseek-r1 and llama3.1, despite English generally having more training data — worth investigating further at scale.
- All three models tend to produce **fluent, confidently-formatted fake hadiths** (correct narrator-citation style, sometimes even fake source citations like `[Sahih Bukhari 1.780]`) rather than declining to answer — this is the core hallucination risk the benchmark is built to surface.

> This is a 6-hadith pilot meant to prove the pipeline works end-to-end. Conclusions are directional, not statistically robust — the full benchmark (1000 hadiths/book × 6 books = 6000 hadiths per model/language) is needed for reliable per-book, per-model comparisons.

## 6. Requirements

```
pandas
nltk
sentence-transformers
requests
tqdm
numpy
```

Install with:
```bash
pip install -r requirements.txt
```

Also requires [Ollama](https://ollama.com) running locally (`ollama serve`) with the target models pulled (e.g. `ollama pull mistral`).

## 7. Running the pipeline

```bash
python3 01_sample_data.py
python3 02_run_llm.py --model mistral:7b --lang english
python3 02_run_llm.py --model mistral:7b --lang arabic
python3 03_evaluate.py --file results/mistral:7b_english.csv
python3 03_evaluate.py --file results/mistral:7b_arabic.csv
python3 04_results_summary.py
```

Repeat steps 2–3 for each model (`deepseek-r1:8b`, `llama3.1:8b`, ...) and language, then run `04_results_summary.py` once at the end to combine everything.
