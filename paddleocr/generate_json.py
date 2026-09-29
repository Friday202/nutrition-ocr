import json

from paddleocr import PaddleOCR 
import common.helpers as helpers 
import logging
from tqdm import tqdm
from pathlib import Path
import pandas as pd

import os

from common.helpers import create_folder, get_data

# ocr = PaddleOCR(lang="en") # Uses English model by specifying language parameter
# ocr = PaddleOCR(ocr_version="PP-OCRv4") # Uses other PP-OCR versions via version parameter
# ocr = PaddleOCR(device="gpu") # Enables GPU acceleration for model inference via device parameter
# ocr = PaddleOCR(
#     text_detection_model_name="PP-OCRv5_mobile_det",
#     text_recognition_model_name="PP-OCRv5_mobile_rec",
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# ) # Switch to PP-OCRv5_mobile models

"""
Runs the paddle ocr over train dataframe generating json for every image. 
EDit thsi file as preprocessing. 
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def extract_ocr_text(result) -> str:
    if not result:
        return ""
    # result is a list of predictions, one per image
    single = result[0]
    texts = single.get("rec_texts", [])
    texts = [t for t in texts if t and t.strip()]
    return " ".join(texts)

if __name__ == "__main__":
    ocr = PaddleOCR(    
        #text_detection_model_name="PP-OCRv5_server_det",
        #text_recognition_model_name="latin_PP-OCRv3_mobile_rec",  
        #use_doc_unwarping=False, 
        #use_textline_orientation=True, 
        text_det_limit_side_len=1600,
        text_det_thresh = 0.3,
        text_det_box_thresh = 0.5,
        text_rec_score_thresh = 0.5,
        device="gpu", 
        lang="sl"        
    )      

    df = helpers.get_nutris_train_dataframe()
    log.info(f"Loaded dataframe: {len(df)} rows")

    base_path = helpers.get_img_folder_path("nutris")

    device = "gpu"

    log.info(f"Loading PaddleOCR (device={device}) ...")   

    #output_dir = Path("./paddle_ocr_jsons_comparinson")
    #output_dir.mkdir(parents=True, exist_ok=True)    

    img = "007.jpg"

    from pathlib import Path

    output_dir = Path("paddle_ocr_visualizations")
    output_dir.mkdir(exist_ok=True)

    for image_path, ground_truth in get_data("demo"):
        if os.path.basename(image_path) != img:
            continue

        result = ocr.predict(input=image_path)

        for res in result:
            res.save_to_img(str(output_dir))

        print("Extracted Text:", result)
        print("Ground Truth:", ground_truth)

    exit(0)
    
    out_path = Path("paddle_ocr_best_train_new_2.jsonl")

    with out_path.open("a", encoding="utf-8") as f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="OCR"):
            image_path = str(base_path / row["FileName"])

            if not Path(image_path).exists():
                log.warning(f"Image not found, skipping: {image_path}")
                continue

            gt = row["Ingredients"]
            if pd.isna(gt) or not str(gt).strip():
                print(f"Ground truth is empty, skipping: {image_path}")  
                continue

            prediction = ocr.predict(input=image_path)

            record = {
                "image_path": image_path,
                "ground_truth": gt,
                "prediction": extract_ocr_text(prediction)
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            #for res in result:
                #res.save_to_json(save_path=output_dir)                
            
    log.info(f"Done generating .json files.")
  