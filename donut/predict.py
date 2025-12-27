from transformers import DonutProcessor, VisionEncoderDecoderModel
from datasets import load_from_disk
import torch
from PIL import Image


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
    model = VisionEncoderDecoderModel.from_pretrained(dataset_type + "-model" + checkpoint_path)

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


if __name__ == "__main__":
    # Runs prediction on test set from his dataset not arbitrary image
    data_type = "sroie"

    # Load processor and model, move model to device
    processor = get_processor(data_type)
    model = get_model(data_type)
    device = get_device()

    check_model_processor_compatibility(model, processor)
    model.to(device)

    # Grab the first sample from the processed test section of dataset
    test_sample = get_processed_dataset(data_type)["test"][0]

    # Run prediction
    prediction, target = run_prediction(test_sample, model, processor, device)
    print("Prediction:", prediction)
    print("Target:", target)
