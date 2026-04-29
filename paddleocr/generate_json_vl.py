from paddleocr import PaddleOCRVL 
import common.helpers as helpers 
import logging
from tqdm import tqdm
from pathlib import Path
import pandas as pd

"""
Runs the paddle ocr vl over train dataframe
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

if __name__ == "__main__":
    ocr = PaddleOCRVL()   

    df = helpers.get_nutris_train_dataframe()
    log.info(f"Loaded dataframe: {len(df)} rows")

    base_path = helpers.get_img_folder_path("nutris")

    device = "gpu"

    log.info(f"Loading PaddleOCR (device={device}) ...")   

    output_dir = Path("./paddle_ocr_jsons_vl")
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
  