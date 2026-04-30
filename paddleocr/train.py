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

import json
import logging
import os
import random
from pathlib import Path

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
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"   # swap to 3B if you have VRAM
DATA_PATH = "paddle_ocr_jsons/ocr_results.jsonl"
OUTPUT_DIR = "models/ocr-corrector"
MAX_SEQ_LEN = 512
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
# remaining 0.1 = internal test (separate from the official donut test set)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen2 attention
)

SFT_ARGS = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    eval_strategy="steps",        # 'evaluation_strategy' renamed in newer trl
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",             # swap to "wandb" if you want tracking
    dataloader_num_workers=4,
    max_length=MAX_SEQ_LEN,   # moved here from SFTTrainer
    dataset_text_field="text",    # moved here from SFTTrainer
    packing=False,
)


# ------------------------------------------------------------------ #
# Prompt template
# ------------------------------------------------------------------ #
SYSTEM_PROMPT = (
    "You are a food label OCR corrector. "
    "You receive raw OCR text extracted from a Slovenian food product image. "
    "Your task is to extract and clean ONLY the ingredients list from it. "
    "Output only the cleaned ingredients string, nothing else."
)

def make_prompt(ocr_text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{ocr_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def make_full_example(ocr_text: str, gt_ingredients: str) -> str:
    """Full prompt + completion for SFT."""
    return make_prompt(ocr_text) + gt_ingredients + "<|im_end|>"


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #
def load_data(path: str):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
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

    def to_hf_dataset(records):
        return Dataset.from_dict({
            "text": [make_full_example(r["ocr_text"], r["gt_ingredients"]) for r in records]
        })

    return to_hf_dataset(train), to_hf_dataset(val)


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #
def main():
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

    train_dataset, val_dataset = load_data(DATA_PATH)

    trainer = SFTTrainer(
        model=model,
        args=SFT_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    log.info("Starting training ...")
    trainer.train()

    log.info(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ #
# SLURM template — save as slurm_finetune.sh
# ------------------------------------------------------------------ #
"""
#!/bin/bash
#SBATCH --job-name=ocr-corrector
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/finetune_%j.log

source activate your_env
python finetune.py
"""