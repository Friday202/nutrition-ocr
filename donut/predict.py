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

    if not has_target:
        return prediction, None

    # load reference target
    target = processor.token2json(sample["target_sequence"])  # What is here target_sequence?
    return prediction, target


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
        if isinstance(d, dict) and "ingredients" in d:
            return [
                item.get("text", "").strip()
                for item in d.get("ingredients", [])
                if isinstance(item, dict) and "text" in item
            ]
        return None

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


if __name__ == "__main__":
    # Runs prediction on test set from his dataset not arbitrary image
    data_type = "nutris-slim"
    checkpoint_path = ""

    # Load processor and model, move model to device
    processor = get_processor(data_type)
    model = get_model(data_type, checkpoint_path=checkpoint_path)
    device = get_device()

    check_model_processor_compatibility(model, processor)
    model.to(device)

    rows = []

    # Grab the first sample from the processed test section of dataset
    dataset = get_processed_dataset(data_type)["test"]
    for i in range(len(dataset)):
        test_sample = dataset[i]

        # Run prediction
        prediction, target = run_prediction(test_sample, model, processor, device)

        target = parse_ingredients(target)
        prediction = parse_ingredients(prediction)

        wer = helpers.compute_wer(target, prediction)
        cer = helpers.compute_cer(target, prediction)

        print("Target:", target)
        print("Prediction:", prediction)
        print("-" * 50)

        rows.append({
            "WER": wer,
            "CER": cer
        })

    df = pd.DataFrame(rows)
    df.to_csv("ocr_eval_results.csv", index=False)
