"""
finetune.py

LoRA fine-tune a small LLM (default: Qwen2.5-1.5B-Instruct) to correct
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
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct" 
DATA_PATH = "paddle_ocr_best_train.jsonl"
OUTPUT_DIR = "qwen_model_14B_lr2e4r8_params_la16_G_Enhanced/ocr-corrector"
MAX_SEQ_LEN = 1024
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
# remaining 0.1 = internal test (separate from the official donut test set)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen2 attention
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

SFT_ARGS = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    #gradient_accumulation_steps=4,
    learning_rate=2e-4,
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
    max_seq_length=MAX_SEQ_LEN,   # moved here from SFTTrainer
    dataset_text_field="text",    # moved here from SFTTrainer
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


#SYSTEM_PROMPT = (
#    "You are a food label OCR corrector. "
#    "You receive raw OCR text extracted from a Slovenian food product image. "
#    "Your task is to extract and clean ONLY the ingredients list from it. "
#    "Output only the cleaned ingredients string, nothing else."
#)

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
# Data loading
# ------------------------------------------------------------------ #
def tokenize_and_mask(example, tokenizer, max_len=1024):
    prompt = make_prompt(example["ocr_text"])
    completion = example["gt_ingredients"] + "<|im_end|>"

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(prompt + completion, add_special_tokens=False,
                           max_length=max_len, truncation=True)["input_ids"]

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # If truncated, labels must match input length
    labels = labels[:len(full_ids)]

    return {"input_ids": full_ids, "attention_mask": [1]*len(full_ids), "labels": labels}

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

    print(f"\n{split_name}")
    print(f"  Total examples: {len(records)}")
    print(f"  > {max_len} tokens: {len(too_long)} ({100 * len(too_long) / len(records):.2f}%)")
    print(f"  Mean: {np.mean(lengths):.1f}")
    print(f"  Max:  {np.max(lengths)}")

    print("  Percentiles:")
    for p in [50, 75, 90, 95, 99, 100]:
        print(f"    p{p:>3}: {np.percentile(lengths, p):.1f}")

    if too_long:
        print("\nFirst few long prompts:")
        for idx, n in too_long[:10]:
            print(f"  example {idx}: {n} tokens")

def load_data(path: str, limit: int | None = None):
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

    def to_hf_dataset2(records):
        return Dataset.from_dict({
            "text": [make_full_example(r["ocr_text"], r["gt_ingredients"]) for r in records]
        })
    
    def to_hf_dataset(records):
        raw = Dataset.from_dict({
            "ocr_text":       [r["prediction"]       for r in records],
            "gt_ingredients": [r["ground_truth"] for r in records],
        })
        return raw.map(
            lambda ex: tokenize_and_mask(ex, tokenizer),
            remove_columns=["ocr_text", "gt_ingredients"],
        )
    
    check_prompt_lengths(train, tokenizer, "train")
    check_prompt_lengths(val, tokenizer, "val")

    quit()

    return to_hf_dataset(train), to_hf_dataset(val)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
    global tokenizer
    log.info(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    train_dataset, val_dataset = load_data(DATA_PATH, limit=None)

    """
    trainer = SFTTrainer(
        model=model,
        args=SFT_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    """

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
    main()


