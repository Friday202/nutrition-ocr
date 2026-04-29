from paddleocr import PaddleOCR 
import common.helpers as helpers 
import logging
from tqdm import tqdm
from pathlib import Path
import pandas as pd

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
Runs the paddle ocr over train dataframe
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if __name__ == "__main__":
    ocr = PaddleOCR(    
        use_doc_orientation_classify=False,
        use_doc_unwarping=False, 
        use_textline_orientation=False, 
        device="gpu", 
        lang="sl"
    )      

    df = helpers.get_nutris_train_dataframe()
    log.info(f"Loaded dataframe: {len(df)} rows")

    base_path = helpers.get_img_folder_path("nutris")

    device = "gpu"

    log.info(f"Loading PaddleOCR (device={device}) ...")   

    output_dir = Path("./paddle_ocr_jsons")
    output_dir.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="OCR"):
        image_path = str(base_path / row["FileName"])

        if not Path(image_path).exists():
            log.warning(f"Image not found, skipping: {image_path}")            
            continue

        gt = row["Ingredients"]
        if pd.isna(gt) or not str(gt).strip():            
            continue

        result = ocr.predict(input=image_path)
        for res in result:
            res.save_to_json(save_path=output_dir)                
            
    log.info(f"Done generating .json files.")
  