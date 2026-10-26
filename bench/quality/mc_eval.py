"""Multiple-choice accuracy, scored the same way on every stack.

Each question is shown to the model with no examples, and the model must reply with a letter
(A, B, C or D). We look at which letter it finds most likely as the next token. This only needs
`adapter.next_token_topk_batch`, so Hugging Face, vLLM and llama.cpp are all scored identically.
(EleutherAI's lm-evaluation-harness can't score llama.cpp reliably. scripts/run_lm_eval.sh
is an optional extra check for HF and vLLM.)

Tasks: ARC-Challenge (full test set), MMLU (20 questions per subject, random but fixed),
and HellaSwag (1000 questions, random but fixed).
Our numbers won't match public leaderboards because the prompts are different. What
matters is the difference between configs.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from ..stats import binomial_stderr

LETTERS = "ABCDEFGH"
INSTR = "Answer the following multiple choice question. Reply with only the letter of the correct option.\n\n"


@dataclass
class Item:
    text: str        # the full question with its answer options
    n_choices: int
    gold: int        # index of the correct answer


def _fmt(question: str, choices: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(choices))
    return f"{INSTR}{question}\n{opts}\nAnswer:"


def build_items(task: str, seed: int = 0, mmlu_per_subject: int = 20, hellaswag_n: int = 1000,
                limit: int | None = None) -> list[Item]:
    from datasets import load_dataset

    rng = random.Random(seed)
    items: list[Item] = []
    if task == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        for r in ds:
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                continue
            items.append(Item(_fmt(r["question"], r["choices"]["text"]), len(labels), labels.index(r["answerKey"])))
    elif task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
        by_subject: dict[str, list] = {}
        for r in ds:
            by_subject.setdefault(r["subject"], []).append(r)
        for subj in sorted(by_subject):
            rows = by_subject[subj]
            rng.shuffle(rows)
            for r in rows[:mmlu_per_subject]:
                items.append(Item(_fmt(r["question"], r["choices"]), 4, int(r["answer"])))
    elif task == "hellaswag":
        ds = load_dataset("Rowan/hellaswag", split="validation")
        rows = list(ds)
        rng.shuffle(rows)
        for r in rows[:hellaswag_n]:
            q = "Which ending most plausibly continues the text?\n\n" + r["ctx"]
            items.append(Item(_fmt(q, r["endings"]), 4, int(r["label"])))
    else:
        raise ValueError(task)
    return items[:limit] if limit else items


def encode(tokenizer, item: Item) -> list[int]:
    chat = tokenizer.apply_chat_template([{"role": "user", "content": item.text}],
                                         add_generation_prompt=True, tokenize=False)
    return tokenizer(chat, add_special_tokens=False).input_ids


def evaluate(adapter, tokenizer, items: list[Item], batch: int = 64) -> dict:
    prompts = [encode(tokenizer, it) for it in items]
    preds, no_letter = [], 0
    for i in range(0, len(prompts), batch):
        for it, top in zip(items[i:i + batch], adapter.next_token_topk_batch(prompts[i:i + batch])):
            scores = [float("-inf")] * it.n_choices
            for text, lp in top:
                t = text.strip()
                if len(t) == 1 and t in LETTERS[:it.n_choices]:
                    j = LETTERS.index(t)
                    scores[j] = max(scores[j], lp)
            if max(scores) == float("-inf"):
                no_letter += 1   # none of the answer letters were among the likely tokens
                preds.append(-1)
            else:
                preds.append(max(range(it.n_choices), key=scores.__getitem__))
    acc = sum(p == it.gold for p, it in zip(preds, items)) / len(items)
    return {"acc": acc, "stderr": binomial_stderr(acc, len(items)), "n": len(items),
            "no_letter_frac": no_letter / len(items), "preds": preds}
