import cv2
import pytesseract
from common.helpers import create_folder
import numpy as np


def preprocess(image, debug=False):
    if isinstance(image, str):
        image = cv2.imread(image)

    image = image.copy()
    step = 0

    if debug:
        # Save initial image to debug
        cv2.imwrite(f"debug/{step}_initial.jpg", image)

    # Resize to enlarge small text
    scale_percent = 200  # 200% size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        step += 1
        cv2.imwrite(f"debug/{step}_resized.jpg", image)

    # Invert colors only if the mean pixel value indicates a dark background
    # mean_brightness = np.mean(image)
    # if mean_brightness > 127:
    #    image = cv2.bitwise_not(image)
    #    if debug:
    #        step += 1
    #        cv2.imwrite(f"debug/{step}_inverted.jpg", image)

    # Blur the image to reduce noise
    image = cv2.medianBlur(image, 5)
    if debug:
        step += 1
        cv2.imwrite(f"debug/{step}_blurred.jpg", image)

    # Apply adaptive thresholding (better for uneven lighting)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )
    if debug:
        step += 1
        cv2.imwrite(f"debug/{step}_thresh.jpg", image)

    return image


def show_bounding_box(ocr_data, img):
    image = img.copy()
    for i in range(len(ocr_data["text"])):
        text = str(ocr_data["text"][i])  # <-- make sure it's a string
        conf = int(ocr_data["conf"][i]) if ocr_data["conf"][i] != '' else -1

        if text.strip() != "" and text.lower() != "nan" and conf > 0:
            x, y, w, h = (
                ocr_data["left"][i],
                ocr_data["top"][i],
                ocr_data["width"][i],
                ocr_data["height"][i],
            )

            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put text + confidence above box
            cv2.putText(
                image,
                f"{text} ({conf})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(f"debug/bounding_boxes.jpg", image)


def run_prediction(image, debug=False, config=None):  # Either image path or image array
    if debug:
        create_folder("debug", flush=True, parent_path="")

    pre_processed_image = preprocess(image, debug)

    if config is None:
        config = r'--oem 2 --psm 6 -c preserve_interword_spaces=1 -c tessedit_write_images=true'

    ocr_data = pytesseract.image_to_data(
        pre_processed_image,
        lang="slv",
        config=config,
        output_type=pytesseract.Output.DATAFRAME
    )

    if debug:
        show_bounding_box(ocr_data, pre_processed_image)

    return postprocess(ocr_data)


def postprocess(ocr_data):
    ocr_data = ocr_data[ocr_data["text"].notnull() & (ocr_data["text"].str.strip() != "")]
    ocr_data = ocr_data[ocr_data["conf"] > 30]  # Filter out low-confidence results
    ocr_text = " ".join(ocr_data["text"].tolist())

    return ocr_text
