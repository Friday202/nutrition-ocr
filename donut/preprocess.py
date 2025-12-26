from datasets import load_from_disk
from pathlib import Path
from transformers import DonutProcessor
from datasets import load_dataset
import torch
import json
import pandas as pd

import config


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


def create_json_meta_data_file_xslx(overwrite=False):
    metadata_path = Path(config.DATA_DIR).joinpath("key")
    image_path = Path(config.DATA_DIR).joinpath("img")

    # First check if metadata.jsonl already exists
    metadata_file = image_path.joinpath('metadata.jsonl')
    if metadata_file.exists():
        print(f"[INFO] '{metadata_file}' already exists.")
        if not overwrite:
            print("[INFO] Skipping creation (set overwrite=True to replace).")
            return
        else:
            print("[INFO] Overwriting existing file...")

    metadata_list = []
    df = pd.read_excel(metadata_path / "__IzvozIngredients.xlsx")
    print(df.columns.tolist())

    for _, row in df.iterrows():
        file_stem = str(row["FileName"]).strip()
        file_stem = Path(file_stem).stem  # ensures no .jpg duplication
        text = str(row["Ingredients"]).strip()

        image_file = image_path / f"{file_stem}.jpg"

        if image_file.is_file():
            metadata_list.append({
                "text": json.dumps({"text_sequence": text}, ensure_ascii=False),
                "file_name": f"{file_stem}.jpg"
            })
        else:
            print(f"[WARNING] Image file '{image_file}' does not exist. Skipping.")

    with open(metadata_file, 'w', encoding='utf-8') as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

    print("[INFO] metadata.jsonl created successfully.")
    exit(0)


def create_json_meta_data_file(overwrite=False):
    metadata_path = Path(config.DATA_DIR).joinpath("key")
    image_path = Path(config.DATA_DIR).joinpath("img")

    # First check if metadata.jsonl already exists
    metadata_file = image_path.joinpath('metadata.jsonl')
    if metadata_file.exists():
        print(f"[INFO] '{metadata_file}' already exists.")
        if not overwrite:
            print("[INFO] Skipping creation (set overwrite=True to replace).")
            return
        else:
            print("[INFO] Overwriting existing file...")

    metadata_list = []
    for file_name in metadata_path.glob("*.json"):
        with open(file_name, "r", encoding="utf-8") as json_file:
            # load json file
            data = json.load(json_file)
            # create "text" column with json string
            text = json.dumps(data)
            # add to metadata list if image exists
            if image_path.joinpath(f"{file_name.stem}.jpg").is_file():
                metadata_list.append({"text": text, "file_name": f"{file_name.stem}.jpg"})

    with open(image_path.joinpath('metadata.jsonl'), 'w') as outfile:
        for entry in metadata_list:
            json.dump(entry, outfile)
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


def preprocess():
    create_json_meta_data_file_xslx(False)
    image_path = Path(config.DATA_DIR).joinpath("img")

    # Load dataset
    dataset = load_dataset("imagefolder", data_dir=image_path, split="train")

    # Tokenize dataset
    new_special_tokens = []  # new tokens which will be added to the tokenizer
    task_start_token = "<s>"  # start of task token
    eos_token = "</s>"  # eos token of tokenizer

    # proc_dataset = dataset.map (
    #     lambda sample: preprocess_documents_for_donut(sample, new_special_tokens, task_start_token, eos_token)
    # )

    # proc_dataset = dataset.map(
    #     lambda sample: preprocess_documents_for_donut(sample, new_special_tokens, task_start_token, eos_token),
    #     batched=False,
    #     batch_size=1
    # )

    # Load processor
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

    proc_dataset = dataset.map(
       lambda batch: preprocess_documents_for_donut_batch(batch, new_special_tokens, task_start_token, eos_token),
       batched=True,
       batch_size=1,
    )

    # Load processor
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")

    # add new special tokens to tokenizer
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]})
    processor.feature_extractor.size = [720, 960]  # should be (width, height)
    processor.feature_extractor.do_align_long_axis = False

    processed_dataset = proc_dataset.map(
        lambda sample: transform_and_tokenize(sample, processor=processor, split="train"),
        remove_columns=["image", "text"]
    )

    # Save processed dataset and processor
    processed_dataset.save_to_disk(config.PROCESSED_DATASET_DIR)
    processor.save_pretrained(config.PROCESSOR_DIR)


if __name__ == "__main__":
    preprocess()
