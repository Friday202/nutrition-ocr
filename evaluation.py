# Reads jsonl results from respective ocr pipeline (tesseract, paddleocr and donut) and performs evaluation and
# visualization
from pathlib import Path
import common.helpers as helpers
from jiwer import wer, cer


if __name__ == "__main__":
    # Scan this directory for ocr results
    base_dir = Path("results")
    jsonl_ocr_results_paths = []

    for item in base_dir.iterdir():
        print(f"Found OCR results in: {item.name}, this will be evaluated.")
        if item.is_dir():
            for jsonl_file in item.glob("*.jsonl"):
                jsonl_ocr_results_paths.append(jsonl_file)

    # Now get ground truth data
    gt_data = helpers.get_demo_data("data_ocr/img/")
    gt_dict = {Path(img_path).stem: label for img_path, label in gt_data}
    print(f"Found {len(gt_dict)} ground truth samples for evaluation.")

    # Evaluate each OCR results jsonl file
    for ocr_results_path in jsonl_ocr_results_paths:
        print(f"Evaluating OCR results from: {ocr_results_path}")
        # Load OCR results
        ocr_results = helpers.load_jsonl(ocr_results_path)

        for entry in ocr_results:
            predicted_text = entry['ocr_text']
            file_name = entry['file_name']
            gt_text = gt_dict.get(file_name, "")
            # Simple evaluation: print comparison
            print(f"File: {file_name}")
            print(f"WER: {wer(gt_text, predicted_text):.4f}, CER: {cer(gt_text, predicted_text):.4f}")
            print("-" * 40)
