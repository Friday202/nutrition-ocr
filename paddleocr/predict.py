from paddleocr import PaddleOCR
import os

from common.helpers import create_folder, get_data


def run_prediction(image_path, debug=False):
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    # Run OCR inference on a sample image
    result = ocr.predict(input=image_path)

    # Visualize the results and save the JSON results
    for res in result:
        if debug:
            res.save_to_img("output")
            # res.save_to_json("output")
    return result


if __name__ == "__main__":
    img = "003.jpg"

    for image_path, ground_truth in get_data("demo"):
        if os.path.basename(image_path) != img:
            continue

        text = run_prediction(image_path, debug=True)
        print("Extracted Text:", text)
        print("Ground Truth:", ground_truth)

