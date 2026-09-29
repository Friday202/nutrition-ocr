import json
import csv
import os

file = "paddle_ocr_best_test.jsonl"
output_file = "test_images.csv"

with open(file, "r", encoding="utf-8") as f_in, \
     open(output_file, "w", newline="", encoding="utf-8") as f_out:

    writer = csv.writer(f_out)

    for line in f_in:
        entry = json.loads(line)
        image_path = entry.get("image_path")

        if image_path:
            writer.writerow([os.path.basename(image_path)])

print(f"Created {output_file}")