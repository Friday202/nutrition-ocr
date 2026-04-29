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


def build_dataset(output_path: str, device: str, debug: bool, limit: int | None, vl_model: bool = True):
    # ------------------------------------------------------------------ #
    # Lazy import so the script can be inspected without paddleocr/helpers
    # ------------------------------------------------------------------ #
    import common.helpers as helpers
    from paddleocr import PaddleOCR, PaddleOCRVL

    df = helpers.get_nutris_train_dataframe()
    log.info(f"Loaded dataframe: {len(df)} rows")

    if limit:
        df = df.head(limit)
        log.info(f"Debug limit applied: {limit} rows")

    base_path = helpers.get_img_folder_path("nutris")

    device = "gpu:0"

    log.info(f"Loading PaddleOCR (device={device}) ...")

    if vl_model:
        ocr = PaddleOCRVL()
    else:
        ocr = PaddleOCR(use_textline_orientation=True, lang="sl", device=device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    written = 0

    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

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

            result = ocr.predict(input=image_path)
            for res in result:
                res.save_to_json(save_path=output_dir)                
            
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
