from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import random

import config
import preprocess


def run_prediction(sample, model, processor, device):
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

    # load reference target
    target = processor.token2json(sample["target_sequence"])
    return prediction, target


def evaluate():
    processed_dataset = preprocess.get_processed_dataset()
    processor = DonutProcessor.from_pretrained(config.PROCESSOR_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(config.MODEL_DIR)

    assert len(processor.tokenizer) == model.config.decoder.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    test_sample = processed_dataset["train"][random.randint(0, len(processed_dataset["train"]) - 1)]
    prediction, target = run_prediction(test_sample, model, processor, device)
    print(f"Reference:\n {target}")
    print(f"Prediction:\n {prediction}")

    x = ["train", "test", "validation"]
    for xx in x:
        for i in range(len(processed_dataset[xx])):
            sample = processed_dataset[xx][i]
            prediction, target = run_prediction(sample, model, processor, device)
            print(f"Reference:\n{target}")
            print(f"Prediction:\n{prediction}")


if __name__ == "__main__":
    evaluate()
