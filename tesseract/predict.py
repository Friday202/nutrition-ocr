import cv2
import pytesseract
import os

from common.helpers import create_folder, get_data
import common.helpers as helpers


def preprocess_(image, debug=False):
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
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if debug:
        step += 1
        cv2.imwrite(f"debug/{step}_resized.jpg", image)

    # TODO: Invert colors missing robust method to know whether to invert or not
    # mean_brightness = np.mean(image)
    # if mean_brightness > 127:
    #    image = cv2.bitwise_not(image)
    #    if debug:
    #        step += 1
    #        cv2.imwrite(f"debug/{step}_inverted.jpg", image)

    # Blur the image to reduce noise
    #image = cv2.medianBlur(image, 5)
    #if debug:
    #    step += 1
    #    cv2.imwrite(f"debug/{step}_blurred.jpg", image)

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


def run_prediction(image, preprocess=True, debug=False, config=None):  # Either image path or image array
    if debug:
        create_folder("debug", flush=True, parent_path="")

    if preprocess:
        pre_processed_image = preprocess_(image, debug)
    else:
        pre_processed_image = cv2.imread(image) if isinstance(image, str) else image

    if config is None:
        config = r'--oem 1 --psm 6 -c preserve_interword_spaces=0 -c tessedit_write_images=true'

    try:
        ocr_data = pytesseract.image_to_data(
            pre_processed_image,
            lang="slv",
            config=config,
            output_type=pytesseract.Output.DATAFRAME
        )
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return ""

    if debug:
        show_bounding_box(ocr_data, pre_processed_image)

    return postprocess(ocr_data)


def postprocess(ocr_data):
    # Ensure text is treated safely as string
    text = ocr_data["text"]

    # Convert everything to string, but keep NaNs separate first
    mask_valid = text.notnull()

    # Only apply .str operations on valid rows
    clean_text = text[mask_valid].astype(str)

    mask_non_empty = clean_text.str.strip() != ""

    # Combine masks correctly
    ocr_data = ocr_data[mask_valid].copy()
    ocr_data = ocr_data.loc[mask_non_empty.index[mask_non_empty]]

    # Confidence filter (safe)
    ocr_data = ocr_data[ocr_data["conf"] > 30]

    ocr_text = " ".join(ocr_data["text"].astype(str).tolist())

    return ocr_text


def set_tesseract_path(tesseract_path="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"):
    if not os.path.isfile(tesseract_path):
        raise ValueError(f"The provided Tesseract path is invalid: {tesseract_path}")
    else:
        print("Using Tesseract executable at:", tesseract_path)

    pytesseract.pytesseract.tesseract_cmd = tesseract_path

import tqdm
# improt path
from pathlib import Path
import pandas as pd
import json 

if __name__ == "__main__":
    # Run this if you want on a single image for testing
    set_tesseract_path()

    img = "007.jpg"

    for image_path, ground_truth in get_data("demo"):
        if os.path.basename(image_path) != img:
            continue
    
        text = run_prediction(image_path, preprocess=False, debug=True)
        print("Extracted Text:", text)
        print("Ground Truth:", ground_truth)

    exit(0)
    
    out_path = Path("ocr_results.jsonl")

    df = helpers.get_nutris_test_dataframe()
    print(f"Loaded dataframe: {len(df)} rows")

    base_path = helpers.get_img_folder_path("nutris")
    print(f"Base image path: {base_path}")    

    with out_path.open("a", encoding="utf-8") as f:
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="OCR"):
           
            image_path = str(base_path / row["FileName"])

            if not Path(image_path).exists():
                print(f"Image not found, skipping: {image_path}")
                continue

            gt = row["Ingredients"]
            if pd.isna(gt) or not str(gt).strip():
                print(f"Ground truth is empty, skipping: {image_path}")        
                continue

            prediction = run_prediction(image_path, preprocess=True, debug=False)
            record = {
                "image_path": image_path,
                "ground_truth": gt,
                "prediction": prediction
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")




