"""
Extractive question answering.

Implemented directly against AutoModelForQuestionAnswering rather than the
"question-answering" pipeline task, which exists in transformers 4.x but was
removed in 5.x. Loading the model directly works on both.
"""

from typing import Dict

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

QA_MODEL_NAME = "deepset/roberta-base-squad2"

# Longest question+context window fed to the model; longer contexts are
# truncated from the context side only.
MAX_SEQ_LENGTH = 384

# Longest answer span, in tokens, considered when scoring start/end pairs.
MAX_ANSWER_TOKENS = 50


def load_qa_model(model_name: str = QA_MODEL_NAME):
    """Load the tokenizer and model for extractive QA."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def answer_question(tokenizer, model, question: str, context: str) -> Dict:
    """
    Extract the best answer span for a question from a context.

    Returns a dict with 'answer' and 'score', matching the shape the
    question-answering pipeline used to return.
    """
    if not question or not context:
        return {"answer": "", "score": 0.0}

    inputs = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)

    # Never extract a span out of the question half of the input.
    sequence_ids = inputs.sequence_ids(0)
    context_mask = torch.tensor(
        [sid != 1 for sid in sequence_ids], dtype=torch.bool
    )
    start_probs = start_probs.masked_fill(context_mask, 0.0)
    end_probs = end_probs.masked_fill(context_mask, 0.0)

    # Score every valid (start, end) pair, then take the best.
    scores = torch.outer(start_probs, end_probs)
    valid = torch.triu(torch.ones_like(scores), diagonal=0)
    valid = torch.tril(valid, diagonal=MAX_ANSWER_TOKENS - 1)
    scores = scores * valid

    best = int(torch.argmax(scores))
    start_idx = best // scores.size(1)
    end_idx = best % scores.size(1)
    score = float(scores[start_idx, end_idx])

    answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    if not answer:
        return {"answer": "No answer found in the provided context.", "score": 0.0}

    return {"answer": answer, "score": score}


if __name__ == "__main__":
    tok, mdl = load_qa_model()
    ctx = (
        "To reset your password, visit the login page and click 'Forgot Password'. "
        "Enter your email address and follow the instructions sent to your inbox."
    )
    for q in ["How do I reset my password?", "What is the capital of France?"]:
        result = answer_question(tok, mdl, q, ctx)
        print(f"Q: {q}\nA: {result['answer']} (score={result['score']:.4f})\n")
