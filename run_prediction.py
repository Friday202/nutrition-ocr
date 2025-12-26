# Script to generate results using a trained model

import pytesseract
import os

import common.helpers as helpers
import tesseract.predict as tesseract
import donut.predict as donut
import paddleocr.predict as paddleocr


def set_tesseract_path(tesseract_path):
    if not os.path.isfile(tesseract_path):
        raise ValueError(f"The provided Tesseract path is invalid: {tesseract_path}")
    else:
        print("Using Tesseract executable at:", tesseract_path)

    pytesseract.pytesseract.tesseract_cmd = tesseract_path


def generate_ocr_results(ocr_type, flush=False, debug=False, only_file=None):
    # Create output folder
    output_results_dir = helpers.create_folder(ocr_type, flush)

    # Run prediction on demo data
    for img_path, label in helpers.get_demo_data():

        if only_file is not None and os.path.basename(img_path) != only_file:
            print("Skipping image:", os.path.basename(img_path), " since you specified only_file=" + only_file)
            continue

        if ocr_type == "tesseract":
            ocr_text = tesseract.run_prediction(img_path, debug=debug)
        elif ocr_type == "donut":
            model, processor, device = donut.load_model_and_processor()
            ocr_text = donut.run_prediction_from_image(img_path, model, processor, device)
            ocr_text = donut.clean_donut_text(ocr_text)  # Optional cleaning step for demo data
        else:
            raise ValueError(f"Unsupported OCR type: {ocr_type}")

        helpers.save_to_json(ocr_text, img_path, output_results_dir, flush=flush)

    # Save all results to a single metadata.jsonl file
    helpers.save_to_jsonl(output_results_dir, flush=flush)


if __name__ == "__main__":
    # Set Tesseract executable path
    set_tesseract_path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

    # Code to generate results thru an interface for all OCR methods
    run_for = ["tesseract"]  #  "donut"]

    subset_images = None  # e.g., ["image1.jpg", "image2.jpg"]

    flush_results = True
    show_debug_imgs = False

    # Generate OCR results
    for method in run_for:
        generate_ocr_results(method, flush=flush_results, debug=show_debug_imgs)
