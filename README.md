# Hadith Hallucination Benchmark

A benchmark pipeline that measures how much local LLMs (run via [Ollama](https://ollama.com)) **hallucinate** when asked to recall specific Hadith texts from reference (Book, Chapter, Hadith Number) alone — instead of quoting the real text, do they confidently invent one?

Dataset: 6 major Hadith collections — **Sahih Bukhari, Sahih Muslim, Sunan Abi Da'ud, Jami' al-Tirmidhi, Sunan an-Nasa'i, Sunan Ibn Majah**.

Models evaluated: **Mistral-7B**, **Gemma3-4B**, **Phi4-mini-3.8B**, **Jais-7B** (Arabic-adapted) and **ALLaM-7B** (Arabic-centric), all running locally via Ollama.

**Headline result:** across 30,000 responses (5 models × 2 languages × 3,000 hadiths), **0% were correct**. No model reproduced the actual hadith text from its reference, and refusals were almost nonexistent (≤0.7%). The models fabricated text instead of saying they didn't know.

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
│   └── sampled_500_{Book}.csv       # stratified sample, 500 hadiths/book, seed=42
├── prompts/
│   ├── prompt_english.txt           # English prompt template
│   └── prompt_arabic.txt            # Arabic prompt template
├── results/
│   ├── {model}_{lang}_evaluated.csv # LLM output + text_match_score, theme_match_score, label
│   └── final_summary.csv            # combined comparison across all models
├── Final result/                    # raw + evaluated CSVs, one folder per model
├── paper/                           # LaTeX source, figures (drawio), results PDFs
├── Submission/
│   └── paper.pdf                    # submitted paper
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
Reads `dataset/all_hadiths_clean.csv`, strips whitespace from the `source` column, and draws a **stratified random sample of 500 hadiths per book** (reproducible with `RANDOM_SEED=42`). Saves one file per book to `data/sampled_500_{Book}.csv` (3,000 hadiths in total). Prints a per-book count as a sanity check.

### Step 2 — `02_run_llm.py`
For a chosen `--model` (e.g. `mistral:7b`) and `--lang` (`english` or `arabic`):
1. Loads the matching prompt template from `prompts/`.
2. For each sampled hadith, fills in `{source}`, `{chapter_no}`, `{chapter}`, `{hadith_no}` — **the model is given only the reference, never the actual text** — and sends it to Ollama's local API (`http://localhost:11434`).
3. Records the model's raw response next to the real ground-truth text (`text_en` / `text_ar`).
4. Uses low-temperature decoding (`temperature=0.2`, `repeat_penalty=1.3`) to reduce degenerate repetition, adds a progress bar (`tqdm`) and a 0.5s delay between calls, and writes `"ERROR"` for any failed call instead of crashing.
5. Saves to `results/{model}_{lang}.csv` with columns:
   `hadith_id, source, chapter_no, hadith_no, chapter, ground_truth_en, ground_truth_ar, llm_response, model, lang`

**Prompt design note:** the prompt does *not* forbid the model from including the chain of narrators (isnad) — since real ground-truth text sometimes includes narrator attribution (e.g. "Narrated Abu Huraira:"), forcing the model to omit it would create an unfair, structurally different comparison.

### Step 3 — `03_evaluate.py`
Scores every row on **two independent axes**, because text-match and theme-match catch different failure modes:

- **Level 1 — BLEU score (text match)**: is the model quoting the *exact words* of the real hadith? Uses `nltk.translate.bleu_score.sentence_bleu` with smoothing, scored 0–1.
- **Level 2 — Cosine similarity (theme match)**: does the model's answer at least mean the same thing, even with different wording? Both texts are embedded with the multilingual sentence-embedding model **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**, and cosine similarity between the two vectors is computed, scored 0–1. This model runs **entirely locally** (downloaded once from Hugging Face, cached, no API calls after that) and works across English and Arabic with one shared model.

Before scoring, responses that are refusals or requests for clarification (e.g. "I don't have access to…", "Please provide the reference", "لا أستطيع مساعدتك") are detected by phrase matching and labeled `refused`. They are not scored against the ground truth.

Every other row is then classified:

| Label | Condition | Meaning |
|---|---|---|
| `correct` | BLEU > 0.4 **and** cosine > 0.8 | Model reproduced the real hadith closely |
| `theme_only` | BLEU < 0.4 **and** cosine > 0.8 | Right general idea, wrong/paraphrased wording |
| `hallucinated` | cosine < 0.5 | Model invented unrelated content |
| `partial` | everything else | Somewhere in between |
| `refused` | refusal phrase detected | Model declined or asked for clarification |

Output: `results/{file}_evaluated.csv` (all original columns + `text_match_score` (BLEU), `theme_match_score` (cosine), `label`), plus a printed summary of label counts and average scores per book.

### Step 4 — `04_results_summary.py`
Combines every `*_evaluated.csv` in `results/` into one table — average BLEU, average cosine, and the % breakdown of correct / theme_only / hallucinated / partial — grouped by model and language. Refused rows are excluded from the score averages. Saves `results/final_summary.csv` and prints it.

## 4. Understanding BLEU vs. Cosine (in plain terms)

- **BLEU** = "did it use the *same words*?" It compares overlapping word sequences (n-grams) between the model's answer and the real hadith. Even a correct paraphrase scores near 0 if the wording differs — it is a strict, literal match.
- **Cosine similarity** = "did it mean the *same thing*?" Both texts are converted into numeric vectors by the embedding model, and the angle between those vectors is measured — a small angle (score near 1) means similar meaning, regardless of exact wording.

**Caveat found during testing:** for Arabic religious text, cosine similarity can be misleadingly high (0.7–0.96) even when the hadith content is completely fabricated, because formulaic phrases common to all hadiths (e.g. "صلى الله عليه وسلم", "قال رسول الله") push the embeddings close together regardless of actual content. This is why BLEU and cosine are used **together**, not in isolation — a hallucinated hadith can look thematically close while being factually invented.

## 5. Final results

**Setup:** 5 models × 2 languages × 3,000 hadiths (500 from each of the 6 collections) = **30,000 responses**. All models ran locally via Ollama:

| Model | Ollama tag | Type |
|---|---|---|
| Mistral-7B | `mistral` | General-purpose |
| Gemma3-4B | `gemma3:4b` | General-purpose |
| Phi4-mini-3.8B | `phi4-mini` | General-purpose |
| Jais-7B | `hf.co/mradermacher/jais-adapted-7b-chat-GGUF:Q4_K_M` | Arabic-adapted |
| ALLaM-7B | `iKhalid/allam:7b` | Arabic-centric |

### Aggregate results (`results/final_summary.csv`)

**% Correct = 0.0% in every row.** % Refused is 0% everywhere except Phi4-mini Arabic (0.7%).

| Model | Lang | Avg BLEU | Avg Cosine | % Theme Only | % Partial | % Hallucinated |
|---|---|---|---|---|---|---|
| Mistral-7B | English | 0.007 | 0.559 | 0.47 | 75.03 | 24.50 |
| Mistral-7B | Arabic | 0.006 | 0.719 | 32.27 | 59.73 | 8.00 |
| Gemma3-4B | English | 0.008 | 0.558 | 0.23 | 74.23 | 25.53 |
| Gemma3-4B | Arabic | 0.019 | 0.736 | 39.83 | 50.80 | 9.37 |
| Phi4-mini-3.8B | English | 0.005 | 0.447 | 0.07 | 41.13 | **58.80** |
| Phi4-mini-3.8B | Arabic | 0.019 | 0.500 | 6.97 | 44.83 | 47.50 |
| Jais-7B | English | 0.007 | 0.535 | 0.03 | 66.10 | 33.87 |
| Jais-7B | Arabic | 0.024 | 0.567 | 9.90 | 54.27 | 35.83 |
| ALLaM-7B | English | 0.006 | 0.518 | 0.27 | 59.30 | 40.43 |
| ALLaM-7B | Arabic | 0.015 | 0.480 | 4.37 | 41.37 | **54.27** |

### % Hallucinated by collection

500 hadiths per cell.

| Collection | Mistral EN | Mistral AR | Gemma3 EN | Gemma3 AR | Phi4-mini EN | Phi4-mini AR | Jais EN | Jais AR | ALLaM EN | ALLaM AR |
|---|---|---|---|---|---|---|---|---|---|---|
| Sahih Bukhari | 33.6 | 6.0 | 28.6 | 15.2 | 31.0 | 37.0 | 39.0 | 48.0 | 40.6 | 46.6 |
| Sahih Muslim | 31.0 | 7.8 | 29.8 | 13.4 | 36.6 | 47.2 | 34.6 | 33.8 | 51.0 | 54.2 |
| Sunan Abi Da'ud | 23.2 | 7.8 | 29.4 | 9.0 | 78.6 | 58.0 | 36.6 | 23.6 | 38.6 | 52.0 |
| Jami' al-Tirmidhi | 22.0 | 12.0 | 23.0 | 7.6 | 71.4 | 60.0 | 30.6 | 57.0 | 37.4 | 66.4 |
| Sunan an-Nasa'i | 13.8 | 7.2 | 18.4 | 7.6 | 62.4 | 41.8 | 26.6 | 33.0 | 37.8 | 46.8 |
| Sunan Ibn Majah | 23.4 | 7.2 | 24.0 | 3.4 | 72.8 | 41.0 | 35.8 | 19.6 | 37.2 | 59.6 |

### Key findings

- **No model achieves exact recall.** 0 of 30,000 responses were labeled `correct`. Average BLEU is between 0.005 and 0.024 in every cell, far below the 0.4 threshold, and this holds for every collection, not just on average.
- **Arabic specialization doesn't help.** Jais-7B and ALLaM-7B do no better than the general-purpose models. ALLaM-7B hallucinates *more* in Arabic (54.3%) than in English (40.4%), and its Arabic rate exceeds its English rate in every collection.
- **Phi4-mini hallucinates the most.** In English its rate is 58.8% overall and above 70% for Abi Da'ud, Tirmidhi and Ibn Majah.
- **Mistral-7B and Gemma3-4B hallucinate the least in Arabic (8–9%).** However, much of their output is `theme_only` (32–40%). As explained in Section 4, this partly reflects formulaic Arabic phrasing that inflates cosine similarity, not real recall.
- **Silent fabrication, not abstention.** Refusals are essentially absent (≤0.7%, only from Phi4-mini in Arabic). The models produce confident, plausible-looking hadiths instead of saying they don't know, which is the more dangerous failure mode.

**Implication:** open-weight 4–8B LLMs used without retrieval are not reliable sources for quoting hadith. Tools for Muslim users should use retrieval-augmented generation over a verified hadith corpus rather than relying on the model's memory.

Full write-up: [`Submission/paper.pdf`](Submission/paper.pdf). Per-model raw and evaluated outputs are in [`Final result/`](Final%20result/).

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

# combine the 6 per-book samples into one input file
python3 -c "import pandas as pd,glob; pd.concat(pd.read_csv(f) for f in sorted(glob.glob('data/sampled_500_*.csv'))).to_csv('data/sampled_500_all.csv', index=False)"

python3 02_run_llm.py --model mistral --lang english --data data/sampled_500_all.csv
python3 02_run_llm.py --model mistral --lang arabic  --data data/sampled_500_all.csv
python3 03_evaluate.py --file results/mistral_english.csv
python3 03_evaluate.py --file results/mistral_arabic.csv
python3 04_results_summary.py
```

Repeat steps 2–3 for each model (`gemma3:4b`, `phi4-mini`, `iKhalid/allam:7b`, `hf.co/mradermacher/jais-adapted-7b-chat-GGUF:Q4_K_M`) and both languages, then run `04_results_summary.py` once at the end to combine everything.
