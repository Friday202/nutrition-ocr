# Script to generate results using a trained model
import os

import common.helpers as helpers
import tesseract.predict as tesseract
import donut.predict as donut
import paddleocr.predict as paddleocr


def generate_ocr_results(ocr_type):
    # Create output folder
    output_results_dir = helpers.create_folder(ocr_type, True)

    # Run prediction on demo data
    for img_path, label in get_data():
        if ocr_type == "tesseract":
            ocr_text = tesseract.run_prediction(img_path, debug=False)
        elif ocr_type == "donut":
            model, processor, device = donut.load_model_and_processor()
            ocr_text = donut.run_prediction_from_image(img_path, model, processor, device)
            ocr_text = donut.clean_donut_text(ocr_text)  # Optional cleaning step for demo data
        elif ocr_type == "paddleocr":
            ocr_text = paddleocr.run_prediction(img_path)  # Not yet supported
        else:
            raise ValueError(f"Unsupported OCR type: {ocr_type}")

        helpers.save_to_json(ocr_text, img_path, output_results_dir, flush=True)

    # Save all results to a single metadata.jsonl file
    helpers.save_to_jsonl(output_results_dir, flush=True)


def get_data():
    # Currently only demo data is supported
    return helpers.get_demo_data()


if __name__ == "__main__":
    # Set Tesseract executable path
    tesseract.set_tesseract_path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    # Code to generate results thru an interface for all OCR methods
    run_for = ["tesseract"]  # "donut"]

    # Generate OCR results
    for method in run_for:
        generate_ocr_results(method)
