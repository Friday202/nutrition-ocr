from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image
import re


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
    target = processor.token2json(sample["target_sequence"])
    return prediction, target


def load_model_and_processor():
    processor = DonutProcessor.from_pretrained("donut_ocr/processor_ocr_ingredients")
    model = VisionEncoderDecoderModel.from_pretrained("donut_ocr/outputs/donut-ocr-ingredients" + "/checkpoint-19000")

    assert len(processor.tokenizer) == model.config.decoder.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model, processor, device


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


def clean_donut_text(prediction_json):
    """
    Extract text_sequence and remove <s> </s> tokens.
    """
    if not isinstance(prediction_json, dict):
        return ""

    text = prediction_json.get("text_sequence", "")

    # Remove <s> and </s> tokens
    text = re.sub(r"</?s>", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text
