# Script to generate results using a trained model

import common.helpers as helpers
import pytesseract
import tesseract_ocr.preditc as tesseract
import donut_ocr.predict as donut


def generate_tesseract_ocr_results():
    # Point to your tesseract installation
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # Create output folder
    output_results_dir = helpers.create_folder("tesseract_ocr", True)

    # Run prediction on demo data
    for img_path, label in helpers.get_demo_data():
        ocr_text = tesseract.run_prediction(img_path, debug=True)
        helpers.save_to_json(ocr_text, img_path, output_results_dir, flush=True)

    # Generate .jsonl file with OCR results for all data - used in common evaluation script
    helpers.save_to_jsonl(output_results_dir, flush=True)


def generate_donut_ocr_results():
    # Create output folder
    output_results_dir = helpers.create_folder("donut_ocr", True)

    # Load model
    model, processor, device = donut.load_model_and_processor()

    # Run prediction on demo data
    for img_path, label in helpers.get_demo_data():
        ocr_text = donut.run_prediction_from_image(img_path, model, processor, device)
        string_text = donut.clean_donut_text(ocr_text)  # this step is optional but for demo data needed
        helpers.save_to_json(string_text, img_path, output_results_dir, flush=True)

    # Generate .jsonl file with OCR results for all data - used in common evaluation script
    helpers.save_to_jsonl(output_results_dir, flush=True)


if __name__ == "__main__":
    # generate_tesseract_ocr_results()
    generate_donut_ocr_results()
