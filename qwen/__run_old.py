from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_name = "Qwen/Qwen2.5-7B-Instruct"
params = "7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("paddle_ocr_best_test.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        img_path = record["image_path"]
        gt = record["ground_truth"]
        ocr_text = record["prediction"]

        prompt = f"Extract only Slovenian ingredients from the following OCR text. Output only the cleaned Slovenian ingredients string with numbers if there are any, nothing else. Output must contain only Slovenian words: {ocr_text}"

        messages = [    
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        with open(f"paddle_ocr_best_test_qwen_{params}.jsonl", "a", encoding="utf-8") as output_file:
            record = {
                "image_path": img_path,
                "ground_truth": gt,
                "prediction": response
            }         
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        

print("DONE")