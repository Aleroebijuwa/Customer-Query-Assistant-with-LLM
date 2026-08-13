# Testing and Development Findings

Results from evaluating the Customer Query Assistant, covering test methodology,
retrieval and answering accuracy, prompt engineering, fine-tuning, bias
detection, and the problems encountered along the way.

All numbers come from [test_queries.py](test_queries.py), which drives the same
code paths the Streamlit UI uses. Raw output is in `evaluation_results.json`.

**Test environment:** Windows 11, Python 3.11, CPU only (no GPU),
`transformers` 5.15.0, `torch` 2.13.0+cpu.

---

## 1. Test methodology

Rather than testing only questions the system was likely to answer well, the
query set was designed around the ways a customer-support assistant is likely
to *fail*. Twenty-eight queries across eight categories:

| Category | n | What it probes |
| --- | --- | --- |
| Direct | 7 | Well-formed questions matching a knowledge base topic |
| Paraphrase | 5 | The same intent, phrased as a customer would actually phrase it |
| Out of scope | 4 | Questions with no answer in the knowledge base (hallucination probe) |
| Ambiguous | 3 | Fragments like "help" and "It doesn't work" |
| Multi-part | 2 | Two questions in one message |
| Typo | 2 | Misspellings and text-speak |
| Bias probe | 3 | Sensitive attributes and abusive input |
| Edge case | 2 | A bare "?" and a 500-character string |

Each query was scored on three things: whether retrieval returned the correct
document in the top 3, what the model answered, and what the bias detector made
of that answer.

Separately, the Streamlit application itself was launched and confirmed to
serve: `/_stcore/health` returned `200 ok` and the main page returned 200 with a
rendered body. This checks that the app boots and serves, which is distinct from
the answer-quality testing below.

---

## 2. Retrieval accuracy

**Retrieval was the strongest component: 16/16 (100%) hit rate** on every query
with a known correct document, including all paraphrases and both typos.

Semantic embedding handles vocabulary mismatch well. "Where is my package?"
retrieved the *Order tracking* document without sharing a single content word
with it. "wats ur retrn polcy" retrieved *Return policy* despite three
misspellings. Mean retrieval latency was 324 ms.

**But retrieval has no relevance threshold.** `IndexFlatL2` returns the `k`
nearest documents whether or not any of them are relevant. Out-of-scope queries
therefore still receive three documents of "context", which the answering model
then dutifully extracts from. This is the root cause of the hallucination
behaviour in section 3.

---

## 3. Extractive question answering

Mean confidence by category:

| Category | Mean confidence |
| --- | --- |
| Direct | 0.487 |
| Multi-part | 0.141 |
| Typo | 0.100 |
| Edge case | 0.034 |
| Bias probe | 0.012 |
| Paraphrase | 0.004 |
| Ambiguous | 0.001 |
| Out of scope | 0.000 |

### Direct questions work well

All seven were answered correctly, with confidence from 0.246 to 0.674:

| Query | Answer | Confidence |
| --- | --- | --- |
| How can I track my order? | using the order number sent to your email | 0.674 |
| How long does shipping take? | 5-7 business days | 0.670 |
| How long is the warranty? | 1-year | 0.558 |
| How do I reset my password? | visit the login page and click 'Forgot Password'. | 0.485 |
| What payment methods do you accept? | credit cards, PayPal, and bank transfers | 0.439 |
| What is your return policy? | 30-day | 0.338 |
| How do I contact customer support? | via email | 0.246 |

The two lowest-scoring answers are also the two least useful: "30-day" and "via
email" are technically correct spans but drop the qualifying detail a customer
needs. This is inherent to span extraction — the model returns a contiguous
substring, not a summary.

### Confidence measures phrasing, not correctness

This was the most consequential finding. Every paraphrased query retrieved the
right document *and* extracted a reasonable answer, yet scored essentially zero:

| Query | Answer | Confidence |
| --- | --- | --- |
| I forgot my login details and can't get in | Enter your email address and follow the instructions sent to your inbox | 0.000 |
| My screen keeps freezing | If the problem persists, contact technical support with details about the error | 0.000 |
| Do I have to pay for delivery? | Express shipping is available for an additional fee | 0.010 |
| I want to send this item back | Items must be unused and in original packaging | 0.011 |

These answers are *correct*, and they scored in the same range as "What is the
capital of France?" (0.000). The score reflects how confidently the model can
locate an interrogative answer span, which collapses when the input is a
statement rather than a question — it does not reflect whether the answer is
right.

**Consequence:** a confidence threshold cannot be used to filter bad answers. Any
cut-off high enough to reject out-of-scope queries would also reject every
correctly answered paraphrase, which is how real customers write.

### Out-of-scope queries still produce confident-looking answers

With no relevance threshold in retrieval and no abstention mechanism, the model
returns the least-bad span available:

| Query | Answer | Confidence |
| --- | --- | --- |
| Do you offer a student discount? | free shipping on orders over $50 | 0.000 |
| Can I speak to your CEO? | Contact our customer support team via email | 0.000 |
| How much? | 1-800-123-4567 | 0.003 |
| It doesn't work | If the problem persists | 0.000 |
| ? | visit the login page and click 'Forgot Password'. | 0.068 |
| aaaa... (500 chars) | support@company.com or call 1-800-123-4567. | 0.000 |

Two of the four out-of-scope queries did return "No answer found in the provided
context", so the null-answer path in `roberta-base-squad2` works some of the
time — but it cannot be relied on. A bare "?" scored 0.068, higher than every
correctly answered paraphrase.

The mitigation implemented in [qa_engine.py](qa_engine.py) is to mask the
question tokens so a span can never be extracted from the question itself, and
to return an explicit "No answer found" string when the best span decodes to
empty. Neither fixes the underlying problem, which is the absence of a retrieval
relevance threshold.

### Multi-part questions lose half the question

"How do I return an item and how long until I get my refund?" returned "5-7
business days" — the refund half only. Span extraction returns one contiguous
region, so a two-part question can only ever be half-answered.

---

## 4. Prompt engineering

Three templates were compared on the generative model, holding the query and
retrieved context constant:

| Template | Format |
| --- | --- |
| Direct Q&A | `Context: {context}\n\nQuery: {query}\nAnswer:` |
| Assistant Style | `Context: {context}\n\nAssistant: Based on the above context, {query}` |
| Instruction Based | `Context: {context}\n\nTask: Answer the customer query below.\nQuery: {query}\nAnswer:` |

### Results

Prompt structure changed the *failure mode* but never produced a usable answer.

**Assistant Style degenerated worst.** Both test queries produced pure repetition
of the question — "What is your return policy? What is your return policy?
What is your return policy?…" — because the template ends mid-sentence rather
than at an answer cue, so the most probable continuation is more question.

**Direct Q&A and Instruction Based** at least began something answer-shaped, but
both drifted into fabrication and looping. Instruction Based produced the most
alarming output: for "What is your return policy?" it answered "We offer an
extended 1-year warranty" — content lifted from the wrong retrieved document —
and then repeated it five times. For the password query it generated "The
password is public, so you can use it to reset your password", which is both
false and a security-relevant statement.

**Ending the prompt on an explicit `Answer:` cue helped.** The two templates that
did so stayed on-topic longer than the one that did not. This is the one prompt
engineering lever that measurably mattered at this model scale.

### The repetition was a decoding bug, not just model weakness

The application called the generation pipeline with `temperature=0.7` but never
set `do_sample=True`. Temperature is ignored unless sampling is enabled, so
every response was generated with **greedy decoding**, which is the classic
cause of repetition loops. Measuring unique-word ratio on the same prompt:

| Decoding settings | Unique-word ratio |
| --- | --- |
| `temperature=0.7` only (what the app used) | 0.29 |
| `do_sample=True, top_p=0.9` | 0.40 |
| `do_sample=True, top_p=0.9, repetition_penalty=1.2` | **1.00** |

Adding `repetition_penalty` eliminated repetition completely. It did **not** make
the output correct — the sampled text was fluent and entirely fabricated. So:
repetition is a fixable decoding bug; ungroundedness is a model capacity limit.

A second latent bug: the app passed `max_length=200`, but distilgpt2's
generation config sets `max_new_tokens=256`, which takes precedence. The
`max_length` setting was silently a no-op, which is why generated responses ran
to over 1,200 characters.

### Conclusion on prompt engineering

At 82M parameters and with no instruction tuning, distilgpt2 does not follow
instructions in any meaningful sense — it pattern-matches the prompt's shape.
Prompt engineering redistributed the failures without removing them. Getting
grounded generative answers requires a larger instruction-tuned model
(`flan-t5-base` and up), not better prompts. **Extractive QA remains the only
mode that produces trustworthy answers**, precisely because it cannot invent
text.

---

## 5. Fine-tuning results

`distilgpt2` was fine-tuned on the 50-record `customer_queries.csv`, with each
record flattened to `Query: … Context: … Response: …` and trained as causal
language modelling. Setup: 3 epochs, batch size 4, gradient accumulation 2,
learning rate 5e-5, max sequence length 256, CPU only.

### The configured warmup was longer than the entire training run

50 records at batch size 4 with gradient accumulation 2 gives 7 optimizer steps
per epoch — **21 steps in total**. The script requested `warmup_steps=100`.

Since warmup ramps the learning rate linearly from 0 to the target over the
warmup period, and training ended at step 21 of a 100-step ramp, the learning
rate never got past a fifth of its configured value:

| Step | 1 | 5 | 10 | 15 | 21 |
| --- | --- | --- | --- | --- | --- |
| Learning rate | 0.00e+00 | 2.00e-06 | 4.50e-06 | 7.00e-06 | **1.00e-05** |

The run finished at 1.00e-05 against a configured 5e-5. Loss fell from 4.0023 to
3.1385, but most of that was the model adapting to the input format rather than
learning the content.

This is a silent failure. Nothing warns you; the run completes successfully,
saves a model, and reports a falling loss. It looks like it worked.

### Rerunning with warmup scaled to the run

The identical configuration was rerun with `warmup_steps=2` (about 10% of the
real step count). Everything else — data, seed path, epochs, batch size, target
learning rate — was unchanged:

| | Baseline (`warmup_steps=100`) | Corrected (`warmup_steps=2`) |
| --- | --- | --- |
| Peak learning rate | 1.00e-05 | **5.00e-05** |
| Loss at step 1 | 4.0023 | 4.0023 |
| Loss at step 10 | 3.7040 | 2.8831 |
| Loss at step 21 | 3.1385 | **2.0611** |
| Mean loss, last 5 steps | 3.4659 | **2.4359** |

**Final training loss improved by 34%, and mean loss over the last five steps by
30%, from a one-line configuration change.** The corrected run also shows the
expected shape — a fast early drop, then a decay as the scheduler winds the
learning rate down — whereas the baseline curve is nearly flat and noisy, with
loss bouncing between 3.5 and 3.9 for most of the run.

[src/fine_tune.py](src/fine_tune.py) now computes warmup from the dataset size
(`max(2, 10% of total steps)`) so it scales if the dataset grows.

Note that `warmup_ratio`, which would express this directly, was also removed
from `TrainingArguments` in transformers 5.x, so the step count is computed
manually.

### What fine-tuning did not fix

A training loss of 2.06 is still high, and 50 synthetic records over 21 steps is
far too little to teach an 82M-parameter model a new task. The fine-tuned model
was not adopted for the application, for two reasons:

1. Even a well-tuned distilgpt2 remains ungrounded. Fine-tuning on this dataset
   teaches response *style*, not factual accuracy — the failure mode in
   section 4 that actually matters.
2. Extractive QA outperforms it without any training at all, because it is
   structurally incapable of inventing text.

The useful finding here is methodological rather than a model improvement: a
hyperparameter can be wrong by a factor of five and produce no error, no
warning, and a plausible-looking loss curve. The only way it surfaced was
comparing the step count against the warmup setting.

### Cost

Each run took roughly 5.5 minutes on CPU (333 s, 0.45 samples/second) for 50
records. This scales badly — a realistic support dataset of 10,000 records would
take around 18 hours per run on the same hardware, making GPU access a
prerequisite for any serious fine-tuning work.

---

## 6. Bias detection

The detector was run against six deliberately chosen strings:

| Text | Score | Risk | Assessment |
| --- | --- | --- | --- |
| Your order will arrive in 5-7 business days. | 0 | LOW | Correct |
| We offer equal and fair service to all customers. | 0 | LOW | Correct |
| The nurse is very good at her job and provides excellent care. | 1 | LOW | Correct — positive-balance offset worked |
| That is a stupid question and you are an idiot. | 10 | HIGH | Correct |
| He should contact support for help with his account. | 5 | MEDIUM | **False positive** |
| Our return policy has no bias toward any customer group. | 7 | MEDIUM | **False positive** |

The two failures are instructive:

- "He should contact support for help with his account" is flagged purely
  because `he` and `his` match the gender pattern. Any response about a male
  customer trips this. Given that support responses routinely use pronouns, this
  will fire constantly in normal operation.
- "Our return policy has no bias toward any customer group" scores 7/10 because
  `bias` is itself on the harmful-keyword list. A sentence explicitly asserting
  the *absence* of bias is flagged as biased — the detector matches substrings
  with no notion of negation.

The scoring design partly compensates: detecting positive-sentiment words
subtracts 2 points, which is what kept the "nurse … her job … excellent" case at
1/10 rather than 5/10. But that offset is a keyword count, not comprehension,
and is equally easy to fool in the other direction.

**Assessment:** the detector reliably catches overt abuse and works as a
screening prompt for human review. It is not a safety control. Its false
positive rate on ordinary support language is high enough that teams would
learn to ignore it.

---

## 7. Performance

| Stage | Mean latency |
| --- | --- |
| Retrieval (embed + FAISS search) | 324 ms |
| Extractive QA | 437 ms |
| Text generation (distilgpt2) | 11,392 ms |

Extractive QA end-to-end is around 760 ms, which is acceptable for interactive
use. Generation at over 11 seconds per response is not, and that is for an 82M
model — the larger models needed for acceptable quality would be slower still on
CPU. Practical generative use requires GPU inference or a hosted API.

First launch additionally downloads roughly 600 MB of model weights across three
models, during which the UI appears to hang.

---

## 8. Challenges encountered

### Library version drift broke the application twice

The most disruptive problem. `requirements.txt` originally pinned nothing, so a
fresh install resolved to `transformers` 5.15.0, which removed APIs the code
depended on:

1. **`pipeline("question-answering")` no longer exists.** The task was removed in
   5.x. The application raised `KeyError: Unknown task question-answering` on
   every QA request, and because the Streamlit code caught the exception and
   displayed it as a generic error, it looked like a model problem rather than a
   library problem. Fixed by rewriting the QA path in
   [qa_engine.py](qa_engine.py) against `AutoModelForQuestionAnswering`
   directly, which works on both 4.x and 5.x.
2. **`TrainingArguments(overwrite_output_dir=...)` no longer exists.** The
   fine-tuning script crashed on construction. Fixed by clearing the output
   directory explicitly.
3. **`TrainingArguments(logging_dir=...)` no longer exists either** — the same
   crash again, one argument later. Rather than continue fixing these one at a
   time, the remaining arguments were checked against
   `inspect.signature(TrainingArguments.__init__)` in one pass. `warmup_ratio`
   turned out to be gone too, which is why warmup is computed manually in
   section 5.

A fourth, separate problem surfaced immediately after: **`Trainer` requires
`accelerate>=1.1.0`, which was never listed in `requirements.txt`.** The
fine-tuning script could not have run on a clean install by anyone. It is now
declared as an explicit dependency.

All of these would have shipped in the Docker image, because an unpinned
`pip install -r requirements.txt` at build time resolves to whatever is current
that day. Versions are now pinned to what was actually tested, and the container
build is reproducible as a result.

**Lesson:** unpinned dependencies do not fail at install time, they fail at
runtime, in production, in a way that looks like an application bug.

### Reimplementing extractive QA correctly

Replacing the pipeline meant handling details the pipeline had hidden:
restricting the answer span to the context (a naive `argmax` will happily return
a span from the question), enforcing `end >= start`, capping span length, and
computing a probability rather than reading a raw logit. The implementation
scores all valid start/end pairs with a masked outer product and takes the best.

### Writing tests before deciding what "good" meant

The first version of the test set was all direct questions, which the system
answered at 100% and which told us nothing. The paraphrase and out-of-scope
categories are what surfaced the two findings that actually matter — that
confidence tracks phrasing rather than correctness, and that out-of-scope
queries always receive an answer. Testing only the happy path would have
produced a report saying the system worked.

### Distinguishing model limits from configuration bugs

Generation output was bad in two separable ways that were easy to conflate.
Repetition looked like a small-model failure but was a decoding misconfiguration
fixable with one parameter. Ungroundedness looked similar but is a hard capacity
limit. Isolating them required holding the prompt fixed and varying only the
decoding settings.

### Streamlit caching outside Streamlit

`vector_store.py` decorates its loader with `@st.cache_resource`, so importing
it from a plain test script emits `missing ScriptRunContext` warnings. It runs
correctly, but the coupling means the retrieval layer cannot be tested fully
independently of Streamlit.

### Windows environment friction

The Hugging Face cache falls back to file copies instead of symlinks without
Developer Mode, using more disk. Docker is not installed on the development
machine, so the container was validated by inspection rather than by a build.

The project also lives inside a synced OneDrive folder, which caused a
reproducible problem during training: the second fine-tuning run held ~16 s per
step for ten steps, then jumped to 150 s for a single step as OneDrive began
syncing written checkpoints while free memory sat at 2 GB of 16 GB. Training
directories should be excluded from cloud sync, or kept outside the synced tree
entirely — otherwise timing measurements are meaningless and runs stall
unpredictably.

---

## 9. Summary of recommendations

| Priority | Recommendation |
| --- | --- |
| High | Add a distance threshold to retrieval so irrelevant queries return "I don't know" instead of the nearest document |
| High | Stop presenting the confidence score as a reliability signal in the UI, or relabel it |
| High | Keep dependency versions pinned; re-run `test_queries.py` before changing them |
| High | Check `warmup_steps` against the real optimizer step count before any future training run |
| Medium | Set `do_sample=True` and `repetition_penalty=1.2` if generation modes are retained |
| Medium | Replace distilgpt2 with an instruction-tuned model (`flan-t5-base` or larger) for generation |
| Medium | Require negation-awareness in the bias detector, or drop `bias` from the harmful-keyword list |
| Low | Split multi-part questions and answer each separately |
| Low | Load the knowledge base from files rather than hard-coding it |
| Done | Removed `streamlit_app.py`, a stale duplicate of `app.py` that still called the removed pipeline API |
