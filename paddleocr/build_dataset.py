"""
build_dataset.py

Runs PaddleOCR over all images in the train dataframe and saves
(ocr_text, gt_ingredients) pairs to a JSONL file for LLM fine-tuning.

Usage:
    python build_dataset.py --output_path data/ocr_gt_pairs.jsonl --workers 4

On HPC with no GPU per node, set --device cpu.
If you have GPU nodes, set --device gpu:0.
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def extract_ocr_text(result) -> str:
    """Flatten PaddleOCR predict() result into a single string."""
    texts = []
    for res in result:
        # PaddleOCR v3+ returns objects with .rec_texts
        if hasattr(res, "rec_texts") and res.rec_texts:
            texts.extend([t for t in res.rec_texts if t and t.strip()])
        # fallback: dict-style
        elif isinstance(res, dict) and "rec_texts" in res:
            texts.extend([t for t in res["rec_texts"] if t and t.strip()])
    return " | ".join(texts)


def build_dataset(output_path: str, device: str, debug: bool, limit: int | None):
    # ------------------------------------------------------------------ #
    # Lazy import so the script can be inspected without paddleocr/helpers
    # ------------------------------------------------------------------ #
    import common.helpers as helpers
    from paddleocr import PaddleOCR

    df = helpers.get_nutris_train_dataframe()
    log.info(f"Loaded dataframe: {len(df)} rows")

    if limit:
        df = df.head(limit)
        log.info(f"Debug limit applied: {limit} rows")

    base_path = helpers.get_img_folder_path("nutris")

    device = "gpu:0"

    log.info(f"Loading PaddleOCR (device={device}) ...")
    ocr = PaddleOCR(use_textline_orientation=True, lang="sl", device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="OCR"):
            image_path = str(base_path / row["FileName"])

            if not Path(image_path).exists():
                log.warning(f"Image not found, skipping: {image_path}")
                skipped += 1
                continue

            gt = row["Ingredients"]
            if pd.isna(gt) or not str(gt).strip():
                skipped += 1
                continue

            try:
                result = ocr.predict(input=image_path)
                ocr_text = extract_ocr_text(result)
            except Exception as e:
                log.warning(f"OCR failed for {image_path}: {e}")
                skipped += 1
                continue

            if not ocr_text.strip():
                log.debug(f"Empty OCR result for {image_path}")
                skipped += 1
                continue

            record = {
                "product_id": int(row["ProductId"]),
                "barcode": str(row["Barcode"]),
                "filename": row["FileName"],
                "ocr_text": ocr_text,
                "gt_ingredients": str(gt).strip(),
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    log.info(f"Done. Written: {written}, Skipped: {skipped}")
    log.info(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="data/ocr_gt_pairs.jsonl")
    parser.add_argument("--device", default="cpu", help="cpu | gpu:0")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process only N rows (for testing)")
    args = parser.parse_args()

    build_dataset(args.output_path, args.device, args.debug, args.limit)
