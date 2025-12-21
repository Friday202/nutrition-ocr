import common.helpers as helpers
import pytesseract
from tesseract_ocr.preditc import run_prediction

if __name__ == "__main__":
    # Point to your tesseract installation
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Create output folder
    output_results_dir = helpers.create_folder("tesseract_ocr", True)

    # Run prediction on demo data
    for img_path, label in helpers.get_demo_data():
        ocr_text = run_prediction(img_path, debug=True)
        helpers.save_to_json(ocr_text, img_path, output_results_dir, flush=True)

    # Generate .jsonl file with OCR results for all data - used in common evaluation script
    helpers.save_to_jsonl(output_results_dir, flush=True)
