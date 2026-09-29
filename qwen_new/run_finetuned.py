"""
eval.py

Loads the fine-tuned Qwen OCR corrector and runs it on internal_test.jsonl.
Prints OCR input, prediction, and GT side by side.
"""

import json
import re
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
PARAMMS = "32B"
BASE_MODEL_ID = f"Qwen/Qwen2.5-{PARAMMS}-Instruct"
MODEL_DIR     = f"qwen_new_model_params_r16_alpha32_lr1e4/model_{PARAMMS}/checkpoint-17500"   # path to the finetuned model (change if needed)
TEST_DATA     = "paddle_ocr_best_test.jsonl"
BATCH_SIZE    = 4
MAX_NEW_TOKENS = 1024
FILE_NAME = f"qwen_paddle_ocr_best_test_results/results_{PARAMMS}_finetuned_new_lora_params.csv"

# IMPORTANT: this MUST match the MAX_LEN used in train.py. The mismatch
# between a 512-token inference cap and a 1024-token training cap was
# responsible for a large chunk of the earlier bad results.
MAX_LEN = 1536

SYSTEM_PROMPT = (
    "You are a precise food label parser. "
    "Follow instructions exactly and output nothing except what is requested."
)

def build_prompt(ocr_text: str) -> str:
    return (
        f"The following text was extracted by OCR from a Slovenian food product label and may contain errors. Identify and output only the corrected ingredients list in Slovenian. "
        f"The ingredients section in Slovenian typically follows the word 'Sestavine:'. Do not use text from other languages (Croatian 'Sastojci', German 'Zutaten', etc.) or from nutrition tables or storage instructions:\n\n{ocr_text}"
    )

# ------------------------------------------------------------------ #
# Keyword-anchored windowing (must match train.py's version)
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
    ingredients list itself. Mirrors train.py exactly — keep both in sync,
    or better, move this into a shared module both scripts import.
    """
    match = KEYWORD_PATTERN.search(ocr_text)

    if match:
        start_char = max(0, match.start() - chars_before)
        end_char = min(len(ocr_text), match.start() + ocr_token_budget * chars_after_multiplier)
        candidate = ocr_text[start_char:end_char]
    else:
        candidate = ocr_text

    token_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]

    if len(token_ids) <= ocr_token_budget:
        return candidate, False

    token_ids = token_ids[:ocr_token_budget]
    truncated_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return truncated_text, True


def compute_prompt_overhead_tokens(tokenizer) -> int:
    template_with_empty_ocr = make_prompt("")
    return len(tokenizer(template_with_empty_ocr, add_special_tokens=False)["input_ids"])


# ------------------------------------------------------------------ #
# Load
# ------------------------------------------------------------------ #
print("Loading tokenizer and model ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tokenizer.padding_side = "left"     # for batch generation
tokenizer.truncation_side = "left"  # safety net: if truncation still fires, eat the
                                     # system-prompt boilerplate at the front, not the
                                     # assistant header / OCR tail near the end

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
        f"<|im_start|>user\n{build_prompt(ocr_text)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# No completion at inference time, so the whole budget (minus a small
# safety margin) goes to template overhead + OCR text.
_OVERHEAD_TOKENS = compute_prompt_overhead_tokens(tokenizer)
_SAFETY_MARGIN = 16
OCR_TOKEN_BUDGET = MAX_LEN - _OVERHEAD_TOKENS - _SAFETY_MARGIN
print(
    f"MAX_LEN={MAX_LEN} | template overhead={_OVERHEAD_TOKENS} tokens | "
    f"=> ocr_token_budget={OCR_TOKEN_BUDGET} tokens"
)


def run_batch(batch_records: list) -> list[str]:
    windowed_texts = [
        extract_relevant_window(r["prediction"], tokenizer, OCR_TOKEN_BUDGET)[0]
        for r in batch_records
    ]
    prompts = [make_prompt(t) for t in windowed_texts]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,      # safety net only — windowing above should make this a no-op
        max_length=MAX_LEN,
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

import csv

output_file = FILE_NAME 

with open(output_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "prediction", "ground_truth"])  # header

    for i in range(0, len(records), BATCH_SIZE):
        batch = records[i: i + BATCH_SIZE]
        preds = run_batch(batch)
        all_predictions.extend(preds)

        for rec, pred in zip(batch, preds):
            writer.writerow([
                rec["image_path"],
                pred,
                rec["ground_truth"]
            ])

            print(f"{'='*70}")
            print(f"PRED : {pred}")
            print(f"GT   : {rec['ground_truth']}")

print(f"\nDone. {len(all_predictions)} predictions total.")
print(f"Saved to {output_file}")
