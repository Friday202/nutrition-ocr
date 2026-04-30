import json
import logging
import os
from os import path
from pathlib import Path
import common.helpers as helpers
 
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

def extract_ocr_text(result) -> str:
    """Flatten PaddleOCR JSON result into a single string."""
    texts = result.get("rec_texts", [])
    texts = [t for t in texts if t and t.strip()]
    return " | ".join(texts)

if __name__ == "__main__":
    json_name = "ocr_results.jsonl"
    output_path = "paddle_ocr_jsons"

    jsonl_file_path = Path(output_path) / json_name    

    # remove old jsonl file if it exists
    if jsonl_file_path.exists():
        log.info(f"Removing existing file: {jsonl_file_path}")
        jsonl_file_path.unlink()        

    df = helpers.get_nutris_train_dataframe()   

    # Open the .jsonl file in append mode or create if it doesn't exist
    with open(jsonl_file_path, 'a', encoding="utf-8") as jsonl_file:
        # Iterate over all json files in the output_path
        for file_name in Path(output_path).glob("*.json"):
            with open(file_name, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

                image_name = data.get("input_path", "")
                key = os.path.basename(image_name) 

                if key in df["FileName"].values:
                    gt = df[df["FileName"] == key]["Ingredients"].values[0]                    
                else:
                    raise ValueError(f"Key {key} not found in DataFrame.")
                
                ocr_text = extract_ocr_text(data)                               

                record = {                   
                    "filename": key,
                    "ocr_text": ocr_text,
                    "gt_ingredients": str(gt).strip()                    
                }
 
                jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")                                      
