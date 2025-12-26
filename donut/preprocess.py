from datasets import load_from_disk
from pathlib import Path
from transformers import DonutProcessor
from datasets import load_dataset
import torch
import json
import pandas as pd
import os
import numpy as np
import config
import common.helpers as helpers
import random


def get_processed_dataset(test=False):
    processed_dataset = load_from_disk(config.PROCESSED_DATASET_DIR)

    if test:
        sample = processed_dataset[0]
        img_list = sample["pixel_values"]
        img_tensor = torch.tensor(img_list, dtype=torch.float32)
        print(f"Image tensor shape: {img_tensor.shape}")

    train_test_split = processed_dataset.train_test_split(
        test_size=config.TEST_SIZE, seed=config.SEED
    )

    train_val_split = train_test_split["train"].train_test_split(
        test_size=config.VAL_SIZE, seed=config.SEED
    )

    # Step 3: Combine all splits into one dict
    processed_dataset = {
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": train_test_split["test"]
    }

    return processed_dataset


def get_processor():
    processor = DonutProcessor.from_pretrained(config.PROCESSOR_DIR)
    return processor


def create_jsons_from_xslx(key_file_path, is_flat=False, is_slim=False):
    df = pd.read_excel(key_file_path / "nutris.xlsx")

    n_slim = 100  # 5000 - Test only for now
    n_rows = len(df)

    if n_rows <= n_slim or not is_slim:
        pass  # use all rows
    else:
        indices = np.linspace(0, n_rows - 1, n_slim, dtype=int)
        df = df.iloc[indices]
        print(f"[INFO] Slim mode: reduced from {n_rows} to {len(df)} rows.")

    for _, row in df.iterrows():
        file = str(row["FileName"]).strip()
        text = row["Ingredients"]

        json_file_name = Path(file).stem + ".json"

        if is_flat:
            # For flat model use flat string
            if pd.isna(text) or str(text).strip().lower() == "nan" or str(text).strip() == "/":
                ingredients = ""
            else:
                ingredients = str(text).strip()
        else:
            # Else split into list
            ingredients = process_ingredients(text)

        # Wrap each ingredient in a dict apparently that is needed
        ingredients_wrapped = [{"text": ing} for ing in ingredients]

        json_data = {"ingredients": ingredients_wrapped}

        with open(key_file_path / json_file_name, 'w', encoding='utf-8') as json_file:
            json.dump(json_data, json_file, ensure_ascii=False, indent=4)


def process_ingredients(text):
    """
    Split text by commas but:
    - Ignore commas inside parentheses
    - Ignore commas that are part of numbers like 3,6%
    - Trim whitespace for each ingredient
    """
    if not text or str(text).strip().lower() in {"nan", "/"}:
        return []

    ingredients = []
    current = []
    open_parens = 0

    for i, char in enumerate(text):
        # Track parentheses
        if char == "(":
            open_parens += 1
        elif char == ")":
            if open_parens > 0:
                open_parens -= 1

        # Check if this comma should be ignored
        if char == ",":
            # Don't split if inside parentheses
            if open_parens > 0:
                current.append(char)
                continue
            # Don't split if part of a number (digit before AND after comma)
            prev_char = text[i - 1] if i > 0 else ""
            next_char = text[i + 1] if i + 1 < len(text) else ""
            if prev_char.isdigit() and next_char.isdigit():
                current.append(char)
                continue
            # Otherwise, it's a real separator
            ingredient = "".join(current).strip()
            if ingredient:
                ingredients.append(ingredient)
            current = []
        else:
            current.append(char)

    # Add last ingredient
    ingredient = "".join(current).strip()
    if ingredient:
        ingredients.append(ingredient)

    return ingredients


# This function should be in helpers as results als generate jsonl files
def create_json_meta_data_file(data_name, overwrite=True):
    image_path = helpers.get_img_folder_path(data_name)
    key_path = helpers.get_key_folder_path(data_name)
    metadata_file = helpers.get_metadata_jsonl_path(data_name)

    # First check if metadata.jsonl already exists
    if metadata_file.exists():
        print(f"[INFO] '{metadata_file}' already exists.")
        if not overwrite:
            print("[INFO] Skipping creation (set overwrite=True to replace).")
            return
        else:
            print("[INFO] Overwriting existing file...")

    json_list = []
    for file_name in key_path.glob("*.json"):
        with open(file_name, "r", encoding="utf-8") as json_file:
            # load json file
            data = json.load(json_file)
            # create "text" column with json string
            text = json.dumps(data, ensure_ascii=False)
            # add to metadata list if image exists
            if image_path.joinpath(f"{file_name.stem}.jpg").is_file():
                json_list.append({"text": text, "file_name": f"{file_name.stem}.jpg"})

    with open(metadata_file, 'w', encoding='utf-8') as outfile:
        for entry in json_list:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')


"""
def tokenize_dataset(dataset):
    new_special_tokens = []  # new tokens which will be added to the tokenizer
    task_start_token = "<s>"  # start of task token
    eos_token = "</s>"  # eos token of tokenizer

    def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
       
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                        new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                    output += (
                            fr"<s_{k}>"
                            + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                            + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            # excluded special tokens for now
            obj = str(obj)
            if f"<{obj}/>" in new_special_tokens:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj

    def preprocess_documents_for_donut(sample):
        # create Donut-style input
        text = json.loads(sample["text"])
        d_doc = task_start_token + json2token(text) + eos_token
        # convert all images to RGB
        image = sample["image"].convert('RGB')
        return {"image": image, "text": d_doc}

    dataset = dataset.map(preprocess_documents_for_donut)
    return dataset, new_special_tokens, task_start_token, eos_token
"""


def preprocess_documents_for_donut(sample, new_special_tokens, task_start_token, eos_token):
    """
    Convert a dataset sample into Donut-style (image, text) pair.
    """
    text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(text, new_special_tokens) + eos_token
    image = sample["image"].convert("RGB")
    return {"image": image, "text": d_doc}


def preprocess_documents_for_donut_batch(batch, new_special_tokens, task_start_token, eos_token):
    images, texts, outputs = [], [], []

    for img, txt in zip(batch["image"], batch["text"]):
        text_obj = json.loads(txt)
        d_doc = task_start_token + json2token(text_obj, new_special_tokens) + eos_token
        image = img.convert("RGB")
        images.append(image)
        texts.append(d_doc)

    return {"image": images, "text": texts}


def json2token(obj, new_special_tokens, update_special_tokens_for_json_key=True, sort_json_key=True):
    """
    Convert a JSON-like object (dict/list/primitive) into a token sequence string.
    Updates new_special_tokens list with discovered special tokens.
    """
    if isinstance(obj, dict):
        # Shortcut for {"text_sequence": "..."}
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]

        output = ""
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        for k in keys:
            if update_special_tokens_for_json_key:
                start_tok, end_tok = f"<s_{k}>", f"</s_{k}>"
                if start_tok not in new_special_tokens:
                    new_special_tokens.append(start_tok)
                if end_tok not in new_special_tokens:
                    new_special_tokens.append(end_tok)
            output += f"<s_{k}>{json2token(obj[k], new_special_tokens, update_special_tokens_for_json_key, sort_json_key)}</s_{k}>"
        return output

    elif isinstance(obj, list):
        return "<sep/>".join(
            json2token(item, new_special_tokens, update_special_tokens_for_json_key, sort_json_key)
            for item in obj
        )

    else:
        # Primitive value
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # handle categorical special tokens
        return obj


def transform_and_tokenize(sample, processor, split="train", max_length=512, ignore_id=-100):
    # Convert grayscale images to RGB
    image = sample["image"]
    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        pixel_values = processor(
            image, random_padding=(split == "train"), return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return None  # use None, not {}

    # tokenize document
    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id

    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}


def generate_jsons(dataset_type, overwrite=True):
    if "nutris" not in dataset_type:
        return  # Nothing to do sroie already has jsons

    is_flat = "flat" in dataset_type
    if is_flat:
        dataset_type = dataset_type.replace("-flat", "")

    is_slim = "slim" in dataset_type
    if "is_slim":
        dataset_type = dataset_type.replace("-slim", "")

    # Remove previous json files if overwrite is True
    if overwrite:
        key_file_path = helpers.get_key_folder_path(dataset_type)

        for json_file in key_file_path.glob("*.json"):
            os.remove(json_file)
        print(f"[INFO] Removed existing JSON files in '{key_file_path}'.")

        create_jsons_from_xslx(key_file_path, is_flat=is_flat, is_slim=is_slim)
    else:
        print(f"[INFO] JSON files already exist for '{dataset_type}', skipping generation.")

    return dataset_type


def preprocess(dataset_type, debug=False):
    print(f"[INFO] Preprocessing dataset type: {dataset_type}")
    original_name = dataset_type

    # Generate JSON files if needed
    dataset_type = generate_jsons(dataset_type, overwrite=True)

    # Create metadata.jsonl file
    create_json_meta_data_file(data_name=dataset_type, overwrite=True)

    # Load dataset
    image_path = helpers.get_img_folder_path(dataset_type)
    dataset = load_dataset("imagefolder", data_dir=image_path, split="train")

    print(f"[INFO] Dataset has {len(dataset)} images")
    print(f"[INFO] Dataset features are: {dataset.features.keys()}")

    if debug:
        random_sample = random.randint(0, len(dataset)) - 1
        print(f"[DEBUG] OCR text is {dataset[random_sample]['text']}")
        dataset[random_sample]['image'].resize((720, 960)).show()
        quit(0)

    # Tokenize dataset
    new_special_tokens = []  # new tokens which will be added to the tokenizer
    task_start_token = "<s_sl_ingredients>"  # start of task token
    eos_token = "</s>"  # eos token of tokenizer

    # proc_dataset = dataset.map (
    #     lambda sample: preprocess_documents_for_donut(sample, new_special_tokens, task_start_token, eos_token)
    # )

    # Load original processor
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

    proc_dataset = dataset.map(
        lambda batch: preprocess_documents_for_donut_batch(batch, new_special_tokens, task_start_token, eos_token),
        batched=True,
        batch_size=1,
    )

    # Add new special tokens to tokenizer
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})
    processor.feature_extractor.size = [720, 960]  # should be (width, height)
    processor.feature_extractor.do_align_long_axis = True  # False if dataset_type == "sroie" else True

    processed_dataset = proc_dataset.map(
        lambda sample: transform_and_tokenize(sample, processor=processor, split="train"),
        remove_columns=["image", "text"]
    )

    # Save processed dataset and processor
    processed_dataset.save_to_disk(original_name + "-dataset")
    processor.save_pretrained(original_name + "-processor")

    print(f"[INFO] Preprocessing completed. Processed dataset and processor saved.")


if __name__ == "__main__":
    # "nutris" with optional "-slim" / "-flat" or "sroie", slim is 5000 samples full is 23000 samples
    data = "nutris-slim"

    preprocess(data)
