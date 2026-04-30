"""
eval.py

Loads the fine-tuned Qwen OCR corrector and runs it on internal_test.jsonl.
Prints OCR input, prediction, and GT side by side.
"""

import json
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_DIR     = "models/ocr-corrector"
TEST_DATA     = "paddle_ocr_jsons/internal_test.jsonl"
BATCH_SIZE    = 8
MAX_NEW_TOKENS = 256

SYSTEM_PROMPT = (
    "You are a food label OCR corrector. "
    "You receive raw OCR text extracted from a Slovenian food product image. "
    "Your task is to extract and clean ONLY the ingredients list from it. "
    "Output only the cleaned ingredients string, nothing else."
)

# ------------------------------------------------------------------ #
# Load
# ------------------------------------------------------------------ #
print("Loading tokenizer and model ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = "left"  # for batch generation

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base, MODEL_DIR)
model.eval()

# ------------------------------------------------------------------ #
# Data
# ------------------------------------------------------------------ #
records = []
with open(TEST_DATA, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Loaded {len(records)} test records\n")

# ------------------------------------------------------------------ #
# Inference
# ------------------------------------------------------------------ #
def make_prompt(ocr_text: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{ocr_text}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def run_batch(batch_records: list) -> list[str]:
    prompts = [make_prompt(r["ocr_text"]) for r in batch_records]
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    for i, out in enumerate(outputs):
        new_tokens = out[inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(text)
    return results


# ------------------------------------------------------------------ #
# Run and print
# ------------------------------------------------------------------ #
all_predictions = []

for i in range(0, len(records), BATCH_SIZE):
    batch = records[i: i + BATCH_SIZE]
    preds = run_batch(batch)
    all_predictions.extend(preds)

    for rec, pred in zip(batch, preds):
        print(f"{'='*70}")
        print(f"FILE : {rec['filename']}")
        print(f"OCR  : {rec['ocr_text'][:200]}{'...' if len(rec['ocr_text']) > 200 else ''}")
        print(f"PRED : {pred}")
        print(f"GT   : {rec['gt_ingredients']}")

print(f"\nDone. {len(all_predictions)} predictions total.")