import csv

samples_cer_below_2_percent = "samples_cer_below_2_percent_7B.csv"
perfect_recall_image_paths = "internal_test_perfect_fuzzy_recall_image_paths.csv"

# paddle_ocr_best_test_perfect_recall_image_paths, 
# samples_cer_below_2_percent_donut_final.csv, 
# samples_cer_below_2_percent_qwen_final.csv

samples_cer_below_2_percent = "samples_cer_below_6_percent_qwen_final.csv"
# perfect_recall_image_paths = "paddle_ocr_best_test_perfect_fuzzy_recall_image_paths.csv"
perfect_recall_image_paths = "samples_cer_below_6_percent_donut_final.csv"

def read_paths(csv_path: str, column: str = "image_path") -> set[str]:
    paths = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column not in (reader.fieldnames or []):
            raise KeyError(
                f"Column '{column}' not found in {csv_path}. "
                f"Available columns: {reader.fieldnames}"
            )
        for row in reader:
            val = row[column]
            if val:
                paths.add(val.strip())
    return paths


def main():
    cer_paths = read_paths(samples_cer_below_2_percent)
    recall_paths = read_paths(perfect_recall_image_paths)

    overlap = cer_paths & recall_paths

    print(f"Entries in {samples_cer_below_2_percent}: {len(cer_paths)}")
    print(f"Entries in {perfect_recall_image_paths}: {len(recall_paths)}")
    print(f"Overlap (in both): {len(overlap)}")
    if cer_paths:
        print(f"  -> {len(overlap) / len(cer_paths) * 100:.2f}% of CER<2% samples also have perfect recall")
    if recall_paths:
        print(f"  -> {len(overlap) / len(recall_paths) * 100:.2f}% of perfect-recall samples also have CER<2%")


    # Meaning of the samples that have perfect recall 70% of them are also with CER < 2%. 
    only_in_cer = cer_paths - recall_paths
    only_in_recall = recall_paths - cer_paths
    print(f"Only in CER<2% file: {len(only_in_cer)}")
    print(f"Only in perfect-recall file: {len(only_in_recall)}")


if __name__ == "__main__":
    main()


# Conclusion:
# The OCR stage is probably the larger bottleneck, but Qwen still has meaningful room for improvement.
# Qwen has aviable but unsued information 
# Relaxing the CER metric barely helps 
# 70% achieve CER < 2%
# 72% achieve CER < 6%
# 75% achieve CER < 10%