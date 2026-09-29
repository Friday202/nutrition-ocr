from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_from_disk
import torch
from PIL import Image
import ast
import re
from typing import List, Union
import common.helpers as helpers
import pandas as pd


def get_processed_dataset(dataset_type, test_size=0.1, validation_size=0.1, debug=False, seed=42):
    processed_dataset = load_from_disk(dataset_type + "-dataset")

    if debug:
        sample = processed_dataset[0]
        img_list = sample["pixel_values"]
        img_tensor = torch.tensor(img_list, dtype=torch.float32)
        print(f"Image tensor shape: {img_tensor.shape}")

    train_test_split = processed_dataset.train_test_split(
        test_size=test_size, seed=seed
    )

    train_val_split = train_test_split["train"].train_test_split(
        test_size=validation_size, seed=seed
    )

    # Step 3: Combine all splits into one dict
    processed_dataset = {
        "train": train_val_split["train"],
        "validation": train_val_split["test"],
        "test": train_test_split["test"]
    }

    return processed_dataset


def get_processor(dataset_type):
    proc = DonutProcessor.from_pretrained(dataset_type + "-processor")
    return proc


def get_model(dataset_type, checkpoint_path=""):
    if checkpoint_path == "":
        dataset_type += "-model"
    else:
        dataset_type = "outputs/" + dataset_type

    model = VisionEncoderDecoderModel.from_pretrained(dataset_type + checkpoint_path)

    return model


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def check_model_processor_compatibility(model, processor):
    assert len(processor.tokenizer) == model.config.decoder.vocab_size


def run_prediction(sample, model, processor, device, has_target=True):
    # prepare inputs
    pixel_values = torch.tensor(sample["pixel_values"]).unsqueeze(0)
    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids

    # run inference
    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # process output
    prediction = processor.batch_decode(outputs.sequences)[0]
    prediction = processor.token2json(prediction)

    file_name = sample.get("file_name")  

    if not has_target:
        return prediction, None, file_name

    # load reference target, dataset has 3 fields for target: "pixel_values" - the image, "labels" - token ids, "target_sequence" - string
    target = processor.token2json(sample["target_sequence"])
    return prediction, target, file_name

def run_prediction_batch(samples, model, processor, device, has_target=True):
    # Stack pixel values into a batch
    pixel_values = torch.stack([
        torch.tensor(s["pixel_values"]) for s in samples
    ]).to(device)

    task_prompt = "<s>"
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    # Repeat decoder_input_ids for each item in the batch
    decoder_input_ids = decoder_input_ids.repeat(len(samples), 1)

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    results = []
    predictions = processor.batch_decode(outputs.sequences)
    for i, sample in enumerate(samples):
        prediction = processor.token2json(predictions[i])
        file_name = sample.get("file_name")
        target = processor.token2json(sample["target_sequence"]) if has_target else None
        results.append((prediction, target, file_name))
    return results

def run_prediction_from_image(
    image_path,
    model,
    processor,
    device,
    task_prompt="<s>"  # OR "<s_text_sequence>" if you trained that way
):
    # Load image
    image = Image.open(image_path).convert("RGB")

    # Vision preprocessing (THIS replaces dataset preprocessing)
    pixel_values = processor(
        image,
        return_tensors="pt"
    ).pixel_values.to(device)

    # Task prompt
    decoder_input_ids = processor.tokenizer(
        task_prompt,
        add_special_tokens=False,
        return_tensors="pt"
    ).input_ids.to(device)

    # Generate
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        num_beams=1,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # Decode
    sequence = processor.batch_decode(
        outputs.sequences,
        skip_special_tokens=False
    )[0]

    # Tokens → JSON
    prediction = processor.token2json(sequence)

    return prediction


def parse_prediction(prediction):
    # Example parsing function, modify according to your prediction structure
    parsed_output = {}
    if 'ingredients' in prediction:
        parsed_output['ingredients'] = [ingredient['text'] for ingredient in prediction['ingredients']]
    if 'instructions' in prediction:
        parsed_output['instructions'] = prediction['instructions']
    return parsed_output


def parse_ingredients(raw: Union[str, dict]) -> List[str]:
    """
    Parse model output into a list of ingredient strings.
    Falls back gracefully if format is broken.
    """

    def extract_from_dict(d):
        if not isinstance(d, dict) or "ingredients" not in d:
            return None

        ingredients = d["ingredients"]

        if isinstance(ingredients, dict):
            ingredients = [ingredients]

        if not isinstance(ingredients, list):
            return None

        return [
            item.get("text", "").strip()
            for item in ingredients
            if isinstance(item, dict) and "text" in item
        ]

    # Case 1: already a dict
    if isinstance(raw, dict):
        result = extract_from_dict(raw)
        if result:
            return result

    # Case 2: string → try safe eval
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            result = extract_from_dict(parsed)
            if result:
                return result
        except Exception:
            pass

        # Case 3: try to auto-close brackets/braces
        try:
            fixed = raw.strip()
            if fixed.count("{") > fixed.count("}"):
                fixed += "}"
            if fixed.count("[") > fixed.count("]"):
                fixed += "]"

            parsed = ast.literal_eval(fixed)
            result = extract_from_dict(parsed)
            if result:
                return result
        except Exception:
            pass

        # Case 4: regex fallback (last resort)
        matches = re.findall(r"'text'\s*:\s*'([^']+)'", raw)
        if matches:
            return [m.strip() for m in matches]

        # Case 5: total failure → return original string
        return [raw.strip()]

    # Final fallback
    return [str(raw)]

import os 
if __name__ == "__main__":
    # Runs prediction on test set from his dataset not arbitrary image
    # data_type = "nutris-slim"
    # checkpoint_path = "/checkpoint-24000"

    #data_type = "nutris-slim-10000"
    data_type = "nutris-flat-original-size"

    checkpoint_path = ""

    # Load processor and model, move model to device
    processor = get_processor(data_type)
    model = get_model(data_type, checkpoint_path=checkpoint_path)
    device = get_device()

    check_model_processor_compatibility(model, processor)
    model.to(device)

    rows = []
    BATCH_SIZE = 0

    data_type = "final_test"  

    if data_type == "final_test":
        # Load from paddle_ocr_best_test.jsonl
        import json 
        with open("paddle_ocr_best_test.jsonl", "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f]
    else:
        # Grab the first sample from the processed test section of dataset
        dataset = get_processed_dataset(data_type)["test"]

    if BATCH_SIZE > 1:
        print(f"Running batch prediction with batch size {BATCH_SIZE}...")
        for batch_start in range(0, len(dataset), BATCH_SIZE):
            batch = [dataset[i] for i in range(batch_start, min(batch_start + BATCH_SIZE, len(dataset)))]
            print(f"Processing samples {batch_start + 1}–{batch_start + len(batch)} of {len(dataset)}")

            results = run_prediction_batch(batch, model, processor, device)
            for prediction, target, file_name in results:
                if data_type != "sroie":
                    target = parse_ingredients(target)
                    prediction = parse_ingredients(prediction)
                rows.append({"prediction": prediction, "target": target, "file_name": file_name})
    else:
        print("Running single-sample prediction...")
        for i in range(len(dataset)):
            print("Processing sample", i + 1, "of", len(dataset))
            test_sample = dataset[i]

            if data_type == "final_test":                          
                image_path = test_sample["image_path"]
                file_name = image_path
                filename = image_path.replace("\\", "/").split("/")[-1]
                cluster_path = "/shared/workspace/laspp/jakob_petek/data_ocr/nutris/img/" + filename

                print(f"Running prediction for image: {cluster_path}")                

                prediction = run_prediction_from_image(
                    cluster_path, model, processor, device
                )

                target = test_sample["ground_truth"]
            else:
                # Run prediction
                prediction, target, file_name = run_prediction(test_sample, model, processor, device)

            if i % 50 == 0:
                print("Sample prediction " + str(i) + ":", prediction)
                print("Sample target " + str(i) + ":", target)

            if data_type != "sroie":
                target = parse_ingredients(target)
                prediction = parse_ingredients(prediction)

            rows.append({
                "prediction": prediction,
                "target": target,
                "file_name": file_name
            })

    df = pd.DataFrame(rows)
    name = data_type + "_eval_results.csv"
    df.to_csv(name, index=False)
    print(f"Saved predictions and targets to {name}")

