# Customer Query Assistant

A Streamlit application that answers customer-support questions against a small
knowledge base, using retrieval-augmented generation (RAG) over Hugging Face
transformer models. It retrieves relevant FAQ documents with a FAISS vector
index, answers from that retrieved context, and screens every response through a
rule-based bias detector before displaying it.

---

## Contents

- [How it works](#how-it-works)
- [Features](#features)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [How to run](#how-to-run)
- [Testing](#testing)
- [Evaluation summary](#evaluation-summary)
- [Known limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

---

## How it works

```
User query
    |
    v
[ Retrieval ]  all-MiniLM-L6-v2 embeds the query; FAISS IndexFlatL2
               returns the top-k nearest FAQ documents
    |
    v
[ Answering ]  Extractive QA (roberta-base-squad2) selects a span from the
               retrieved context, OR generative (distilgpt2) continues a
               templated prompt
    |
    v
[ Bias check ] Regex + keyword scan assigns a 0-10 risk score
    |
    v
Response + confidence + retrieved sources shown in the UI
```

The knowledge base is 10 built-in support documents defined in
[vector_store.py](vector_store.py) (password resets, order tracking, returns,
billing, shipping, warranty, technical support, and so on). Embeddings are built
on first launch and cached for the session.

---

## Features

**Three answering modes**, selectable in the sidebar:

| Mode | Model | Behaviour |
| --- | --- | --- |
| Question Answering | `deepset/roberta-base-squad2` | Extracts a verbatim span from the context. Cannot invent text. |
| Text Generation | `distilgpt2` | Free-form continuation of a prompt. |
| RAG Assistant | `distilgpt2` + retrieval | Generation with retrieved documents prepended. |

**Retrieval-augmented generation.** A "Use RAG System" toggle prepends the top-k
retrieved documents to the model input. `k` is adjustable from 1 to 5. Retrieved
documents are shown in an expander so every answer can be traced to its source.

**Three prompt templates** — Direct Q&A, Assistant Style, and Instruction
Based — which change how the query and context are formatted before they reach
the model. Additional templates, including few-shot and structured-output
variants, are available in [prompt_engineering.py](prompt_engineering.py).

**Bias detection.** Every response is scanned by
[bias_detector.py](bias_detector.py) across six sensitive-attribute categories
(gender, race, age, ability, religion, politics) plus a harmful-keyword list.
Responses scoring at or above an adjustable threshold are flagged with a
LOW/MEDIUM/HIGH risk level and an explanation of what triggered the flag.

**Confidence scores.** Question Answering mode reports the model's probability
for the selected answer span.

**Conversation history.** Queries, responses, retrieved documents, and bias
analyses are kept for the session and browsable in an expandable list.

**Manual context mode.** With RAG disabled, Question Answering accepts a
pasted block of context to search within instead of the knowledge base.

---

## Project structure

| File | Purpose |
| --- | --- |
| [app.py](app.py) | Streamlit UI and request flow. Application entry point. |
| [qa_engine.py](qa_engine.py) | Extractive QA against `AutoModelForQuestionAnswering`. |
| [vector_store.py](vector_store.py) | Knowledge base, embeddings, and FAISS index. |
| [bias_detector.py](bias_detector.py) | Rule-based bias and harmful-content scoring. |
| [prompt_engineering.py](prompt_engineering.py) | Prompt template builders. |
| [generate_dataset.py](generate_dataset.py) | Builds the synthetic `customer_queries.csv`. |
| [src/fine_tune.py](src/fine_tune.py) | Fine-tunes `distilgpt2` on the dataset. |
| [model_loader.py](model_loader.py) | Standalone demo of loading a seq2seq model. |
| [test_queries.py](test_queries.py) | Evaluation harness across 28 test queries. |
| [FINDINGS.md](FINDINGS.md) | Test results, prompt engineering analysis, fine-tuning results, challenges. |
| [Dockerfile](Dockerfile) | Container build definition. |

---

## Requirements

- Python 3.9 or later (tested on 3.11)
- Roughly 2 GB of disk for model weights, plus about 2 GB for dependencies
- An internet connection on first run, to download models from Hugging Face

Dependency versions are pinned in [requirements.txt](requirements.txt). See
[Known limitations](#known-limitations) for why.

## How to run

### Option 1: Local Python environment

```bash
# 1. Clone and enter the project
git clone https://github.com/Aleroebijuwa/Customer-Query-Assistant-with-LLM.git
cd Customer-Query-Assistant-with-LLM

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

The app opens at http://localhost:8501.

### Option 2: Docker

```bash
docker build -t customer-query-assistant .
docker run -p 8501:8501 customer-query-assistant
```

Then open http://localhost:8501.

Model weights are downloaded on first use and stored at `/app/models` inside the
container (`HF_HOME`). That directory is not a volume, so a fresh container
re-downloads them. To persist the cache between runs:

```bash
docker run -p 8501:8501 -v hf-cache:/app/models customer-query-assistant
```

> **First run is slow.** Three models totalling roughly 600 MB are fetched on
> demand. The container reports itself unhealthy until this finishes; the
> healthcheck allows a 120-second grace period.

---

## Testing

The evaluation harness exercises the same retrieval, answering, and bias code
paths the UI uses, across 28 queries spanning eight categories — direct
questions, paraphrases, out-of-scope questions, ambiguous fragments, multi-part
questions, typos, bias probes, and edge cases:

```bash
python test_queries.py
```

Results are written to `evaluation_results.json`, and the analysis is written up
in [FINDINGS.md](FINDINGS.md).

Individual modules are runnable on their own for quick checks:

```bash
python qa_engine.py          # extractive QA smoke test
python vector_store.py       # retrieval smoke test
python bias_detector.py      # bias scoring smoke test
python prompt_engineering.py # print each prompt template
```

---

## Evaluation summary

Measured on the 28-query test set. Full detail in [FINDINGS.md](FINDINGS.md).

| Metric | Result |
| --- | --- |
| Retrieval hit rate (correct document in top 3) | 16/16 (100%) |
| Mean QA confidence, direct questions | 0.487 |
| Mean QA confidence, paraphrased questions | 0.004 |
| Mean QA confidence, out-of-scope questions | 0.000 |
| Retrieval latency | ~324 ms |
| Extractive QA latency | ~437 ms |
| Generation latency (distilgpt2, CPU) | ~11.4 s |

Retrieval was the strongest component and extractive QA was reliable on directly
phrased questions. Generation quality was the weakest part of the system — see
below.

### Fine-tuning

`distilgpt2` was fine-tuned on the 50-record dataset via
[src/fine_tune.py](src/fine_tune.py):

```bash
python src/fine_tune.py
```

The run takes about 5.5 minutes on CPU and writes to `models/fine_tuned_model`
plus a loss curve at `fine_tune_log.json`. The original configuration requested
`warmup_steps=100` for a run that only performs 21 optimizer steps, so the
learning rate never finished ramping and peaked at a fifth of its target.
Scaling warmup to the real step count cut final training loss from 3.14 to 2.06.
The fine-tuned model is **not** used by the application — see
[FINDINGS.md](FINDINGS.md#5-fine-tuning-results) for why.

---

## Known limitations

**Generation mode produces unusable output.** `distilgpt2` is an 82M-parameter
model with no instruction tuning. In testing it ignored the retrieved context,
looped on repeated phrases, and stated things that were plainly false (for
example, "The password is public"). Text Generation and RAG Assistant modes are
best treated as demonstrations of the pipeline, not as a usable assistant.
**Question Answering mode is the only mode that produces trustworthy answers,**
because extractive QA can only return spans that exist in the retrieved text.

**Confidence scores do not measure correctness.** They measure how closely the
question's phrasing matches the context. Correctly answered paraphrases scored
as low as 0.000 — the same range as out-of-scope questions. Do not use the score
as a correctness filter or an automatic cut-off.

**Out-of-scope questions still get an answer.** Retrieval always returns the top
`k` documents regardless of how poor the match is; there is no relevance
threshold. The extractive model then pulls the least-bad span. "Do you offer a
student discount?" returned "free shipping on orders over $50", and "How much?"
returned a phone number. Confidence was near zero in both cases, but the answer
is still displayed prominently.

**Only one part of a multi-part question is answered.** Extractive QA returns a
single span, so "How do I return an item and how long until I get my refund?"
answered only the refund half.

**The bias detector produces false positives.** It is keyword and regex based,
with no understanding of context. Two examples from testing: "He should contact
support for help with his account" scored MEDIUM (5/10) purely for the pronouns,
and "Our return policy has no bias toward any customer group" scored MEDIUM
(7/10) because the word "bias" is itself on the harmful-keyword list. It should
be read as a prompt for human review, not as a verdict.

**The knowledge base is 10 hard-coded documents.** It covers common support
topics only, and is defined in code rather than loaded from a document store.
Anything outside those 10 topics is out of scope.

**The dataset is synthetic.** `customer_queries.csv` is 50 hand-written
records from [generate_dataset.py](generate_dataset.py), not real customer
traffic, so it does not reflect real-world query distribution or phrasing.

**Dependency versions are pinned deliberately.** `transformers` 5.x removed the
`question-answering` pipeline task. The application now calls
`AutoModelForQuestionAnswering` directly and works on both 4.x and 5.x, but
other unpinned upgrades can break the build the same way. Change the pins in
[requirements.txt](requirements.txt) only after re-running `test_queries.py`.

**CPU only.** No GPU support is configured. Generation takes roughly 11 seconds
per response on CPU, which is too slow for interactive use.

**No persistence.** Conversation history lives in Streamlit session state and is
lost on refresh. The FAISS index is rebuilt on every cold start.

**Not production ready.** There is no authentication, rate limiting, input
sanitisation, or logging.

---

## Troubleshooting

**`KeyError: Unknown task question-answering`** — an older copy of the code
calling the removed pipeline API. Pull the latest revision and reinstall from
the pinned [requirements.txt](requirements.txt).

**Slow or failing first launch** — models are being downloaded from Hugging
Face. Check your connection, and note that roughly 600 MB is fetched.

**`Error retrieving documents`** — `faiss-cpu` or `sentence-transformers` failed
to import. Reinstall with `pip install faiss-cpu sentence-transformers`. Without
them, retrieval falls back to placeholder documents rather than failing loudly.

**Port 8501 already in use** — run on another port with
`streamlit run app.py --server.port=8502`, or map a different host port in
Docker with `-p 8502:8501`.
