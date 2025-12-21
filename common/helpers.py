import os
import shutil
import json
from pathlib import Path


def create_folder(name, flush=False, parent_path="../results/"):
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


def get_demo_data_path():
    return "../data_ocr/img/"


def get_demo_data(path=None):
    # Assuming get_demo_data_path() returns the correct file path
    if path is None:
        data_path = get_demo_data_path() + "metadata.jsonl"
    else:
        data_path = path + "metadata.jsonl"

    # Open the file and iterate through lines
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line as a JSON object
            data = json.loads(line.strip())

            # Extract the text_sequence (it is in the "text" key, which is a JSON string)
            text_data = json.loads(data['text'])
            text_sequence = text_data.get('text_sequence', "")

            # Extract the file_name
            file_name = data.get('file_name', "")
            file_name = os.path.join(get_demo_data_path(), file_name)

            # Yielding pairs of file_name (img) and text_sequence (label)
            yield file_name, text_sequence
