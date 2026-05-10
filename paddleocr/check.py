"""
OCR Quality Scorer
------------------
Reads a JSONL file with fields: filename, ocr_text, gt_ingredients
Scores each sample and outputs a ranked CSV + summary stats.

Usage:
    python ocr_quality_score.py
    python ocr_quality_score.py --data path/to/file.jsonl --out results.csv --threshold-usable 0.6 --threshold-partial 0.2

Dependencies:
    pip install rapidfuzz
"""

import json
import re
import csv
import argparse
from pathlib import Path
from collections import Counter

try:
    from rapidfuzz import fuzz
except ImportError:
    raise SystemExit("Missing dependency: pip install rapidfuzz")


# ─── Config ───────────────────────────────────────────────────────────────────

DATA_PATH = "paddle_ocr_jsons/ocr_results.jsonl"
OUTPUT_CSV = "ocr_quality_results.csv"

THRESHOLD_USABLE  = 0.6   # token recall above this → "usable"
THRESHOLD_PARTIAL = 0.2   # between partial and usable → "partial", below → "garbage"
FUZZY_MATCH_RATIO = 80    # rapidfuzz ratio threshold for near-matches (0–100)


# ─── Scoring helpers ──────────────────────────────────────────────────────────

GARBAGE_PATTERN = re.compile(r"[^\x00-\x7F\u00C0-\u024F\s]")  # non-latin chars


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]  # skip single chars


def token_recall(ocr_text: str, gt: str, fuzzy_ratio: int = FUZZY_MATCH_RATIO) -> float:
    """
    Fraction of GT tokens found in OCR (exact or fuzzy).
    Returns 0.0 if GT is empty.
    """
    gt_tokens = tokenize(gt)
    if not gt_tokens:
        return 0.0

    ocr_tokens = tokenize(ocr_text)
    ocr_lower  = ocr_text.lower()

    found = 0
    for token in gt_tokens:
        if token in ocr_lower:
            found += 1
        elif any(fuzz.ratio(token, t) >= fuzzy_ratio for t in ocr_tokens):
            found += 1

    return found / len(gt_tokens)


def garbage_ratio(text: str) -> float:
    """Fraction of characters that are non-latin/non-ASCII (Cyrillic, symbols, etc.)."""
    if not text:
        return 1.0
    non_latin = len(GARBAGE_PATTERN.findall(text))
    return non_latin / len(text)


def ocr_density(ocr_text: str) -> float:
    """
    Ratio of alphabetic chars to total chars.
    Low density = lots of pipes/spaces/numbers, likely noisy layout.
    """
    if not ocr_text:
        return 0.0
    alpha = sum(c.isalpha() for c in ocr_text)
    return alpha / len(ocr_text)


def classify(recall: float) -> str:
    if recall >= THRESHOLD_USABLE:
        return "usable"
    elif recall >= THRESHOLD_PARTIAL:
        return "partial"
    else:
        return "garbage"


# ─── Main ─────────────────────────────────────────────────────────────────────

def score_file(data_path: str, output_csv: str, fuzzy_ratio: int = FUZZY_MATCH_RATIO):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    rows = []
    skipped = 0

    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [WARN] Line {lineno}: JSON parse error — {e}")
                skipped += 1
                continue

            filename = sample.get("filename", f"line_{lineno}")
            ocr_text = sample.get("ocr_text", "")
            gt        = sample.get("gt_ingredients", "")

            # Scores
            recall  = token_recall(ocr_text, gt, fuzzy_ratio)
            g_ratio = garbage_ratio(ocr_text)
            density = ocr_density(ocr_text)
            label   = classify(recall)

            # Derived flags
            gt_word_count  = len(tokenize(gt))
            ocr_word_count = len(tokenize(ocr_text))
            ocr_empty      = len(ocr_text.strip()) == 0

            rows.append({
                "filename":       filename,
                "label":          label,
                "token_recall":   round(recall, 4),
                "garbage_ratio":  round(g_ratio, 4),
                "ocr_density":    round(density, 4),
                "gt_word_count":  gt_word_count,
                "ocr_word_count": ocr_word_count,
                "ocr_empty":      ocr_empty,
                "gt_ingredients": gt,
                "ocr_text":       ocr_text,
            })

    if not rows:
        print("No valid rows found. Check your JSONL file.")
        return

    # Sort by token_recall descending
    rows.sort(key=lambda r: r["token_recall"], reverse=True)

    # Write CSV
    out_path = Path(output_csv)
    fieldnames = [
        "filename", "label", "token_recall", "garbage_ratio",
        "ocr_density", "gt_word_count", "ocr_word_count",
        "ocr_empty", "gt_ingredients", "ocr_text",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────────────────
    total   = len(rows)
    counts  = Counter(r["label"] for r in rows)
    recalls = [r["token_recall"] for r in rows]
    empties = sum(1 for r in rows if r["ocr_empty"])

    mean_recall   = sum(recalls) / total
    median_recall = sorted(recalls)[total // 2]

    print("\n" + "═" * 58)
    print(f"  OCR Quality Report  ({path.name})")
    print("═" * 58)
    print(f"  Total samples   : {total}  (skipped {skipped} bad lines)")
    print(f"  Empty OCR       : {empties}")
    print()
    print(f"  Token Recall    mean   : {mean_recall:.3f}")
    print(f"  Token Recall    median : {median_recall:.3f}")
    print()
    print(f"  ✅  usable   (≥{THRESHOLD_USABLE})  : {counts['usable']:>4}  ({100*counts['usable']/total:.1f}%)")
    print(f"  ⚠️   partial  ({THRESHOLD_PARTIAL}–{THRESHOLD_USABLE}) : {counts['partial']:>4}  ({100*counts['partial']/total:.1f}%)")
    print(f"  ❌  garbage  (<{THRESHOLD_PARTIAL})  : {counts['garbage']:>4}  ({100*counts['garbage']/total:.1f}%)")
    print()
    print(f"  Output saved to : {out_path.resolve()}")
    print("═" * 58)

    # ── Top 5 / Bottom 5 ──────────────────────────────────────────────────────
    print("\n  Top 5 (best recall):")
    for r in rows[:5]:
        print(f"    [{r['token_recall']:.2f}]  {r['filename']}")
        print(f"           GT : {r['gt_ingredients'][:80]}")

    print("\n  Bottom 5 (worst recall):")
    for r in rows[-5:]:
        print(f"    [{r['token_recall']:.2f}]  {r['filename']}")
        print(f"           GT : {r['gt_ingredients'][:80]}")
    print()

    return rows


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Score OCR quality from a JSONL file.")
    p.add_argument("--data",              default=DATA_PATH,        help="Path to input JSONL")
    p.add_argument("--out",               default=OUTPUT_CSV,       help="Path to output CSV")
    p.add_argument("--threshold-usable",  type=float, default=0.6,  help="Recall ≥ this → usable")
    p.add_argument("--threshold-partial", type=float, default=0.2,  help="Recall ≥ this → partial")
    p.add_argument("--fuzzy-ratio",       type=int,   default=80,   help="rapidfuzz ratio for near-match (0–100)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Allow overriding globals via CLI
    THRESHOLD_USABLE  = args.threshold_usable
    THRESHOLD_PARTIAL = args.threshold_partial

    score_file(
        data_path   = args.data,
        output_csv  = args.out,
        fuzzy_ratio = args.fuzzy_ratio,
    )