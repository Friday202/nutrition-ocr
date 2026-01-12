import os
import shutil
import json
from pathlib import Path
from jiwer import wer, cer
import re
import unicodedata


def create_folder(name, flush=False, parent_path="results/"):
    # Combine parent path with the folder name to get the full folder path
    folder_path = os.path.join(parent_path, name)

    # Check if the folder already exists
    if os.path.exists(folder_path):
        print(f"Folder '{folder_path}' already exists.")
        if flush:
            # Remove all contents of the folder if flush=True
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove subdirectories
                else:
                    os.remove(file_path)  # Remove files
            print(f"Contents of '{folder_path}' have been flushed.")
    else:
        # Create the folder if it doesn't exist
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' has been created.")

    return folder_path


def save_to_json(ocr_text, image_path, output_path, flush=False):
    name = os.path.splitext(os.path.basename(image_path))[0]

    # Check if the json file already exists
    json_file_path = os.path.join(output_path, f"{name}.json")

    if os.path.exists(json_file_path) and flush:
        os.remove(json_file_path)

    # Save OCR text to a JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump({"ocr_text": ocr_text}, json_file, ensure_ascii=False, indent=4)


def save_to_jsonl(output_path, flush=False):
    # First check if metadata.jsonl already exists
    jsonl_file_path = os.path.join(output_path, "metadata.jsonl")
    if os.path.exists(jsonl_file_path) and flush:
        os.remove(jsonl_file_path)

    # Open the .jsonl file in append mode or create if it doesn't exist
    with open(jsonl_file_path, 'a', encoding="utf-8") as jsonl_file:
        # Iterate over all json files in the output_path
        for file_name in Path(output_path).glob("*.json"):
            with open(file_name, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

                # Extract the 'ocr_text' from the JSON data
                ocr_text = data.get("ocr_text", "")

                # Extract the numeric name (e.g., "000" from "000.json")
                file_key = file_name.stem  # this gets the name without extension

                # Prepare the row to be written to the jsonl file
                jsonl_row = {
                    "ocr_text": ocr_text,
                    "file_name": file_key
                }

                # Write the row in JSONL format (one line per entry)
                jsonl_file.write(json.dumps(jsonl_row, ensure_ascii=False) + "\n")


def load_jsonl(jsonl_path):
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list


def get_data_path(data_name):
    here = os.path.abspath(__file__)

    # Go up TWO levels from this file:
    # common → nutrition-ocr → FRI
    out_project_root = os.path.dirname(os.path.dirname(os.path.dirname(here)))

    return os.path.join(out_project_root, "data_ocr", data_name)


def get_img_folder_path(data_name):
    return Path(get_data_path(data_name)).joinpath("img")


def get_key_folder_path(data_name):
    return Path(get_data_path(data_name)).joinpath("key")


def get_metadata_jsonl_path(data_name):
    # Metadata jsonl is in img folder
    return get_img_folder_path(data_name).joinpath("metadata.jsonl")


def get_data(data_name="demo"):
    jsonl_file = get_metadata_jsonl_path(data_name)

    # Open the file and iterate through lines
    with open(jsonl_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object
            data = json.loads(line.strip())

            # Extract the text_sequence (it is in the "text" key, which is a JSON string)
            text_data = json.loads(data['text'])
            text_sequence = text_data.get('text_sequence', "")

            # Extract the file_name
            file_name = data.get('file_name', "")
            file_name = os.path.join(get_img_folder_path(data_name), file_name)

            # Yielding pairs of file_name (img) and text_sequence (label)
            yield file_name, text_sequence


# Add comon evaluation code, CER, WER, substring recall, char precision etc.
def normalize(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^\w\s]", "", s)   # remove punctuation
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _to_text(x):
    if isinstance(x, list):
        return " ".join(x)
    return str(x)


def compute_wer(gt, pred):
    gt_text = normalize(_to_text(gt))
    pred_text = normalize(_to_text(pred))
    return wer(gt_text, pred_text)


def compute_cer(gt, pred):
    gt_text = normalize(_to_text(gt))
    pred_text = normalize(_to_text(pred))
    return cer(gt_text, pred_text)



