import torch

_original_torch_load = torch.load

def torch_load_unsafe(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = torch_load_unsafe


from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, VisionEncoderDecoderModel, EarlyStoppingCallback
from pathlib import Path

import preprocess
import config

import torch
import numpy as np


def train():
    # Grab processed dataset and processor
    processed_dataset = preprocess.get_processed_dataset()
    processor = preprocess.get_processor()

    # Load pre-trained Donut model
    model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

    # Resize embedding layer to match vocabulary size
    new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
    print(f"New embedding size: {new_emb}")

    # Adjust our image size and output sequence lengths
    print(processor.feature_extractor.size)
    size_dict = processor.feature_extractor.size

    # Match encoder image size
    model.config.encoder.image_size = (size_dict["height"], size_dict["width"])

    all_labels = (
            list(processed_dataset["train"]["labels"])
            + list(processed_dataset["validation"]["labels"])
            + list(processed_dataset["test"]["labels"])
    )

    # model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))
    model.config.decoder.max_length = len(max(all_labels, key=len))

    # Add task token for decoder to start
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        num_train_epochs=10,  # TOTAL VALUE
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        save_total_limit=3,
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        overwrite_output_dir=False,
        predict_with_generate=True#,
        #load_best_model_at_end=True,
    )

    # Create Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"]#,
        #eval_dataset=processed_dataset["validation"]#,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train(resume_from_checkpoint=True)  # next day: resume_from_checkpoint=True

    # Save
    local_dir = Path(config.MODEL_DIR)
    local_dir.mkdir(exist_ok=True)

    # Save the processor and model
    processor.save_pretrained(local_dir)
    trainer.save_model(local_dir)
    trainer.create_model_card()


if __name__ == "__main__":
    torch.serialization.add_safe_globals([
        np.ndarray,
        np.core.multiarray._reconstruct,
    ])

    train()
