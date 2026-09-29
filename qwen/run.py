import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# BASE MODEL NOT FINED TUNED 

# ── config ────────────────────────────────────────────────────────────────────
PARAMS      = "32B"
MODEL_NAME  = f"Qwen/Qwen2.5-{PARAMS}-Instruct" # Qwen/Qwen2.5-7B-Instruct"

INPUT_FILE  = "paddle_ocr_best_test.jsonl"
VARIANT     = "G"
OUTPUT_FILE = f"paddle_ocr_best_test_qwen_{PARAMS}_{VARIANT}.jsonl"
MAX_NEW_TOKENS = 1024
BATCH_SIZE  = 8          # tune to your VRAM; single-GPU with lots of VRAM → try 16+

SYSTEM_PROMPT = (
    "You are a precise food label parser. "
    "Follow instructions exactly and output nothing except what is requested."
)

def build_prompt(ocr_text: str) -> str:
    # A - zero shot with context version 1
    #return (
    #    "Extract the ingredients list from the following OCR text from a Slovenian food product label.\n"
    #    "The OCR may contain errors. Output only the corrected Slovenian ingredients string, nothing else.\n\n"
    #    f"OCR text:\n{ocr_text}"
    #)

    # B - zero shot with minimal context
    #return (
    #    f"Extract only the ingredients list from this OCR text of a Slovenian food label. Output only the ingredients string with numbers if present, nothing else:\n{ocr_text}"
    #)

    # C - few shot with context and example 
    #return (
    #    "Extract the ingredients list from OCR text of a Slovenian food label. Output only the corrected ingredients string.\n\n"
    #    "OCR text:\nVoda, sladkor, citronska kislina (16%), E330\n"
    #    "Ingredients: Voda, sladkor, citronska kislina (16%), E330\n\n"
    #    "OCR text:\nmleko posneto 0lio rastlinsko sladkor\n"
    #    "Ingredients: Mleko posneto, olje rastlinsko, sladkor\n\n"
    #    f"OCR text:\n{ocr_text}\nIngredients:"
    #)

    # D - zero shot with context version 2 (general purpose prompt)
    #return (
    #    f"The following text was extracted by OCR from a Slovenian food product label and may contain errors. "
    #    f"Identify and output only the corrected ingredients list in Slovenian. Do not include anything else:\n\n{ocr_text}"
    #)    
    
    # E - zero shot with slovenian context
    #return (
    #    f"Ekstractiraj samo seznam SLOVENSKIH sestavin iz tega OCR besedila. Izhod naj vsebuje samo slovenske sestavine s številkami, če so prisotne, drugače pa nič drugega:\n{ocr_text}"
    #)

    # F - zero shot maximal context
    #return (
    #    f"The following text was poorly extracted by OCR from a Slovenian food product label. "
    #    f"It likely contains character errors, missing spaces, and garbled words. "
    #    f"Reconstruct and output only the complete corrected ingredients list in Slovenian. "
    #    f"Do not output anything else:\n\n{ocr_text}"
    #)

    # G - zero shot maximal context with instructions
    return (
        f"The following text was extracted by OCR from a Slovenian food product label and may contain errors. Identify and output only the corrected ingredients list in Slovenian. "
        f"The ingredients section in Slovenian typically follows the word 'Sestavine:'. Do not use text from other languages (Croatian 'Sastojci', German 'Zutaten', etc.) or from nutrition tables or storage instructions:\n\n{ocr_text}"
    )

# ── model loading ─────────────────────────────────────────────────────────────
print(f"Loading {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "left"          # required for batched causal LM inference

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto",
)
model.eval()

# ── helpers ───────────────────────────────────────────────────────────────────
def records_to_texts(records: list[dict]) -> list[str]:
    """Apply chat template to a batch of records."""
    texts = []
    for rec in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(rec["prediction"])},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return texts

def generate_batch(texts: list[str]) -> list[str]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    generated = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.eos_token_id,
    )

    # strip the prompt tokens from each output
    responses = []
    for input_ids, output_ids in zip(inputs.input_ids, generated):
        new_ids = output_ids[len(input_ids):]
        responses.append(tokenizer.decode(new_ids, skip_special_tokens=True))
    return responses

# ── main loop ─────────────────────────────────────────────────────────────────
records = []
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

# skip already-processed records so you can resume after a crash
done = set()
if Path(OUTPUT_FILE).exists():
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            done.add(json.loads(line)["image_path"])
    print(f"Resuming — {len(done)} records already done.")

todo = [r for r in records if r["image_path"] not in done]

with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo[i : i + BATCH_SIZE]
        texts = records_to_texts(batch)
        responses = generate_batch(texts)

        for rec, response in zip(batch, responses):
            out_f.write(json.dumps({
                "image_path":   rec["image_path"],
                "ground_truth": rec["ground_truth"],
                "prediction":   response,
            }, ensure_ascii=False) + "\n")
        out_f.flush()   # survive a crash mid-run