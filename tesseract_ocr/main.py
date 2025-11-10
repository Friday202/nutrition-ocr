from data import load_data, load_images
from ocr import perform_ocr
from evaluation import cer, wer
from test import compute_likelihood
from nlp import extract_nutrition_info, extract_nutrition_info_generative


def clean_old_results():
    import os
    import glob

    files = glob.glob('../results/*')
    for f in files:
        os.remove(f)


def main():
    clean_old_results()
    #data = load_data("../images/sestavine", "../annotations/sestavine.txt")
    imgs = load_images()


    for img in imgs:
        print("\nProcessing new image...\n")

        detected_text = perform_ocr(img[1])
        print(detected_text)

    exit()


    for annotation, img in data:
        print("Processing new image...")

        detected_text = perform_ocr(img)
        #print(detected_text)
        #nutrition_info = extract_nutrition_info_generative(
       #     "Izlušči slovensko besedilo", detected_text)

        print(detected_text)
        #print(f"{nutrition_info}")
        # compute_likelihood(detected_text, lang="sl")

        # cer_ = cer(detected_text, annotation)
        # wer_ = wer(detected_text, annotation)

        # print(f"Detected text: {detected_text}")
        # print(f"Ground truth: {annotation}")
        # print(f"CER: {cer_:.2%}, WER: {wer_:.2%}")


if __name__ == "__main__":
    main()
