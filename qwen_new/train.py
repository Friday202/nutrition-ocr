"""
finetune.py

LoRA fine-tune a small LLM (default: Qwen2.5-14B-Instruct) to correct
messy PaddleOCR output into clean ingredient strings.

Input:  data/ocr_gt_pairs.jsonl  (built by build_dataset.py)
Output: models/ocr-corrector/

Usage (local):
    python finetune.py

Usage (SLURM):
    sbatch slurm_finetune.sh   # see bottom of this file for template

Dependencies:
    pip install transformers peft trl datasets accelerate bitsandbytes
"""
# Right now we are looking for best paramaters to then train on the same paramaters the 1.5B 3B 7B 14B and 32B model
# Currently we will take the 1k samples and look for best params, once they are found they will be frozen.
# Switch to train dataset of 22k samples and train on the best params for each model size.
# Once traning is done for all run (predict) the 1k test samples and pick best results -> that one goes into final round against donut.
# For the intermidiate test results (22k x 10%) you can run either CER/WER or tit.py

import json
import logging
import os
import random
import re
from pathlib import Path
import numpy as np

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
)
from trl import SFTTrainer, SFTConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Config — edit these
# ------------------------------------------------------------------ #
PARAMS      = "32B"
MODEL_ID = f"Qwen/Qwen2.5-{PARAMS}-Instruct"
DATA_PATH = "paddle_ocr_best_train.jsonl"
OUTPUT_DIR = f"qwen_new_model_params_r16_alpha32_lr1e4/model_{PARAMS}"

# qwen_model_{PARAMS}-new had:
# r=8,
# lora_alpha=16,
# lora_dropout=0.05,
# learning_rate=2e-4,

# --- Sequence-length budget --------------------------------------- #
# MAX_LEN MUST match the value used in run_finetuned.py at inference time.
# A mismatch here was the root cause of the earlier bad results: training
# capped prompt+completion at 1024 tokens while inference truncated the
# input alone at 512, so the model saw a different token budget than it
# was evaluated with.
MAX_LEN = 1536

# Reserve this many tokens for the ground-truth completion when deciding
# how much of the OCR text the prompt is allowed to keep. Adjust upward if
# you see completions getting truncated (check the diagnostic print in
# main()).
COMPLETION_RESERVE = 160

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
# remaining 0.1 = internal test (separate from the official donut test set)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen2 attention
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

SFT_ARGS = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    #gradient_accumulation_steps=4,
    learning_rate=1e-4,
    #lr_scheduler_type="cosine",
    #warmup_steps=2,
    eval_strategy="steps",        # 'evaluation_strategy' renamed in newer trl
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    #metric_for_best_model="eval_loss",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
    logging_steps=10,
    report_to="none",             # swap to "wandb" if you want tracking
    dataloader_num_workers=4,
    max_seq_length=MAX_LEN,       # kept in sync with MAX_LEN above
    dataset_text_field="text",
    packing=False,
)

tokenizer = None


# ------------------------------------------------------------------ #
# Prompt template
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = (
    "You are a precise food label parser. "
    "Follow instructions exactly and output nothing except what is requested."
)

def build_prompt(ocr_text: str) -> str:
    return (
        f"The following text was extracted by OCR from a Slovenian food product label and may contain errors. Identify and output only the corrected ingredients list in Slovenian. "
        f"The ingredients section in Slovenian typically follows the word 'Sestavine:'. Do not use text from other languages (Croatian 'Sastojci', German 'Zutaten', etc.) or from nutrition tables or storage instructions:\n\n{ocr_text}"
    )


def make_prompt(ocr_text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{build_prompt(ocr_text)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def make_full_example(ocr_text: str, gt_ingredients: str) -> str:
    """Full prompt + completion for SFT."""
    return make_prompt(ocr_text) + gt_ingredients + "<|im_end|>"


# ------------------------------------------------------------------ #
# Keyword-anchored windowing
# ------------------------------------------------------------------ #
KEYWORD_PATTERN = re.compile(r"sestavine\s*:?", re.IGNORECASE)


def extract_relevant_window(
    ocr_text: str,
    tokenizer,
    ocr_token_budget: int,
    chars_before: int = 120,
    chars_after_multiplier: int = 6,
):
    """
    Anchor the OCR text window around the Slovenian 'Sestavine:' keyword so
    that, if truncation is still needed to fit ocr_token_budget, it removes
    boilerplate / other-language sections rather than cutting into the
    ingredients list itself.

    Returns (windowed_text, was_hard_truncated).
    Falls back to the raw text (then plain token truncation) if the
    keyword isn't found.
    """
    match = KEYWORD_PATTERN.search(ocr_text)

    if match:
        start_char = max(0, match.start() - chars_before)
        end_char = min(len(ocr_text), match.start() + ocr_token_budget * chars_after_multiplier)
        candidate = ocr_text[start_char:end_char]
    else:
        candidate = ocr_text  # no anchor found; token-level truncation below still applies

    token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]

    if len(token_ids) <= ocr_token_budget:
        return candidate, False

    # Still too long: hard-truncate at the token level, keeping the FRONT of
    # the candidate window (which already starts near the keyword thanks to
    # chars_before), rather than the tail of the original OCR string.
    token_ids = token_ids[:ocr_token_budget]
    truncated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return truncated_text, True


def compute_prompt_overhead_tokens(tokenizer) -> int:
    """Tokens used by the fixed template (system + user boilerplate +
    assistant header) with an empty OCR string — i.e. everything that isn't
    the variable OCR text."""
    template_with_empty_ocr = make_prompt("")
    return len(tokenizer(template_with_empty_ocr, add_special_tokens=False)["input_ids"])


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #
def tokenize_and_mask(example, tokenizer, ocr_token_budget: int, max_len: int):
    windowed_ocr, _was_truncated = extract_relevant_window(
        example["ocr_text"], tokenizer, ocr_token_budget
    )

    prompt = make_prompt(windowed_ocr)
    completion = example["gt_ingredients"] + "<|im_end|>"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    full_ids = prompt_ids + completion_ids

    # Defensive check: if this example still doesn't fit (e.g. an
    # unusually long ground-truth completion), flag it for dropping instead
    # of silently producing an all-masked (-100) label array that
    # contributes zero gradient while still counting toward "trained on N
    # examples" and toward eval_loss.
    dropped = len(full_ids) > max_len

    if dropped:
        # Keep shapes valid so datasets.map doesn't choke; filtered out below.
        return {
            "input_ids": full_ids[:max_len],
            "attention_mask": [1] * min(len(full_ids), max_len),
            "labels": [-100] * min(len(full_ids), max_len),
            "dropped": True,
        }

    labels = [-100] * len(prompt_ids) + completion_ids
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "dropped": False,
    }

import csv

def check_prompt_lengths(records, tokenizer, split_name, max_len=1024):
    lengths = []
    too_long = []

    for i, r in enumerate(records):
        prompt = make_prompt(r["prediction"])
        n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

        lengths.append(n_tokens)

        if n_tokens > max_len:
            too_long.append((i, n_tokens))

    # Write per-example token lengths to CSV
    csv_path = f"{split_name}_prompt_lengths.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "n_tokens"])
        for i, n_tokens in enumerate(lengths):
            writer.writerow([i, n_tokens])

    print(f"\n{split_name}")
    print(f"  Total examples: {len(records)}")
    print(f"  > {max_len} tokens: {len(too_long)} ({100 * len(too_long) / len(records):.2f}%)")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Max:  {np.max(lengths)}")
    print(f"  Saved lengths to: {csv_path}")

    print("  Percentiles:")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"    p{p:>3}: {np.percentile(lengths, p):.1f}")

    if too_long:
        print("\nFirst few long prompts:")
        for idx, n in too_long[:10]:
            print(f"  example {idx}: {n} tokens")


def check_prompt_lengths1(records, tokenizer, split_name, max_len=1024):
    lengths = []
    too_long = []

    for i, r in enumerate(records):
        prompt = make_prompt(r["prediction"])
        n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

        lengths.append(n_tokens)

        if n_tokens > max_len:
            too_long.append((i, n_tokens))

    print(f"\n{split_name} (unwindowed, diagnostic only)")
    print(f"  Total examples: {len(records)}")
    print(f"  > {max_len} tokens: {len(too_long)} ({100 * len(too_long) / len(records):.2f}%)")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Max:  {np.max(lengths)}")

    print("  Percentiles:")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"    p{p:>3}: {np.percentile(lengths, p):.1f}")

    if too_long:
        print("\nFirst few long prompts (will be windowed around 'Sestavine:' before training):")
        for idx, n in too_long[:10]:
            print(f"  example {idx}: {n} tokens")


def load_data(path: str, ocr_token_budget: int, max_len: int, limit: int | None = None):
    records = []

    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break

            line = line.strip()

            if line:
                records.append(json.loads(line))

    random.seed(42)
    random.shuffle(records)

    n = len(records)
    n_train = int(n * TRAIN_SPLIT)
    n_val = int(n * VAL_SPLIT)

    train = records[:n_train]
    val = records[n_train: n_train + n_val]
    test = records[n_train + n_val:]

    log.info(f"Split — train: {len(train)}, val: {len(val)}, internal test: {len(test)}")

    # Save internal test split for eval_paddle.py
    test_path = Path(path).parent / "internal_test.jsonl"
    with open(test_path, "w", encoding="utf-8") as f:
        for r in test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"Internal test set saved to {test_path}")

    def to_hf_dataset(records, split_name):
        raw = Dataset.from_dict({
            "ocr_text":       [r["prediction"]   for r in records],
            "gt_ingredients": [r["ground_truth"] for r in records],
        })
        mapped = raw.map(
            lambda ex: tokenize_and_mask(ex, tokenizer, ocr_token_budget, max_len),
            remove_columns=["ocr_text", "gt_ingredients"],
        )
        n_before = len(mapped)
        mapped = mapped.filter(lambda ex: not ex["dropped"])
        n_dropped = n_before - len(mapped)
        if n_dropped:
            log.warning(
                f"{split_name}: dropped {n_dropped}/{n_before} examples "
                f"({100 * n_dropped / n_before:.2f}%) that still exceeded "
                f"max_len={max_len} even after windowing around 'Sestavine:'."
            )
        return mapped.remove_columns(["dropped"])

    check_prompt_lengths(train, tokenizer, "train", max_len=max_len)
    check_prompt_lengths(val, tokenizer, "val", max_len=max_len)

    quit()

    return to_hf_dataset(train, "train"), to_hf_dataset(val, "val")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    global tokenizer
    log.info(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    overhead_tokens = compute_prompt_overhead_tokens(tokenizer)
    ocr_token_budget = MAX_LEN - overhead_tokens - COMPLETION_RESERVE
    log.info(
        f"MAX_LEN={MAX_LEN} | template overhead={overhead_tokens} tokens | "
        f"completion reserve={COMPLETION_RESERVE} tokens | "
        f"=> ocr_token_budget={ocr_token_budget} tokens"
    )
    if ocr_token_budget < 100:
        log.warning(
            "ocr_token_budget is very small — consider raising MAX_LEN or "
            "lowering COMPLETION_RESERVE."
        )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    train_dataset, val_dataset = load_data(
        DATA_PATH, ocr_token_budget=ocr_token_budget, max_len=MAX_LEN, limit=None
    )

    trainer = SFTTrainer(
        model=model,
        args=SFT_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100),
    )

    log.info("Starting training ...")
    trainer.train()

    log.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log.info("Done.")


if __name__ == "__main__":
    print(f"Starting to fine tune model: {MODEL_ID}")
    main()
