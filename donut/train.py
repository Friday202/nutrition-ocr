from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, VisionEncoderDecoderModel, EarlyStoppingCallback
from transformers.trainer_utils import get_last_checkpoint

import predict as donut
import common.helpers as helpers

import torch
import numpy as np
import os


def train(data_type, start_over=False):
    print("[INFO] Starting training for type:", data_type)

    output_dir = "outputs/" + data_type
    save_dir = data_type + "-model"

    helpers.create_folder(output_dir, flush=start_over, parent_path="")

    last_checkpoint = None
    if not start_over:
        last_checkpoint = get_last_checkpoint(output_dir)

    # Grab processed dataset and processor
    processed_dataset = donut.get_processed_dataset(data_type)
    processor = donut.get_processor(data_type)

    if last_checkpoint is not None:
        print("[INFO] Resuming training from checkpoint:", output_dir)

        model = VisionEncoderDecoderModel.from_pretrained(last_checkpoint)
    else:
        print("[INFO] Starting training from scratch.")

        # Load pre-trained Donut model
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

        # Resize embedding layer to match vocabulary size
        new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer))
        print(f"[INFO] New embedding size: {new_emb}")

        # Adjust our image size and output sequence lengths
        print("[INFO] Feature extractor size:", processor.feature_extractor.size)
        size_dict = processor.feature_extractor.size

        # Match encoder image size
        print("[INFO] Setting model encoder image size to:", size_dict)
        model.config.encoder.image_size = (size_dict["height"], size_dict["width"])

        all_labels = (
                list(processed_dataset["train"]["labels"])
                + list(processed_dataset["validation"]["labels"])
                # + list(processed_dataset["test"]["labels"])
        )

        # model.config.decoder.max_length = len(max(processed_dataset["train"]["labels"], key=len))
        model.config.decoder.max_length = len(max(all_labels, key=len))
        model.generation_config.max_length = model.config.decoder.max_length

        # Add task token for decoder to start
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(['<s>'])[0]

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        lr_scheduler_type="linear",  # default try "cosine"
        warmup_ratio=0.1,   # 10% warmup
        num_train_epochs=10,  # TOTAL VALUE, in reality we see no improvement after 5-6 epochs
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=8,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=True, 
        dataloader_num_workers=4, 
        logging_steps=100,
        save_total_limit=10,
        evaluation_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        overwrite_output_dir=False,
        predict_with_generate=True,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=3,
                early_stopping_threshold=0.01
            )
        ]
    )

    print("[INFO] Starting training, resuming from checkpoint:", last_checkpoint is not None)
    trainer.train(resume_from_checkpoint=last_checkpoint)

    # Save
    trainer.save_model(save_dir)
    trainer.create_model_card()


if __name__ == "__main__": 
    dataset_type = "nutris-slim-5000"

    train(dataset_type)
