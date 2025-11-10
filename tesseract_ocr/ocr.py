import cv2
import pytesseract
import easyocr

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def perform_ocr(image, config=None, id=0):
    pre_processed_image = perform_preprocess(image)

    if id == 0:
        return tesseract_ocr(pre_processed_image, config)

    else:
        return easyocr_ocr(pre_processed_image)


def easyocr_ocr(pre_processed_image):
    reader = easyocr.Reader(['sl'])
    result = reader.readtext(pre_processed_image)
    lines = [text for (_, text, _) in result]
    raw_text_with_lines = "\n".join(lines)
    print(raw_text_with_lines)


def tesseract_ocr(pre_processed_image, config=None):
    if config is None:
        config = r'--oem 2 --psm 6 -c preserve_interword_spaces=1 -c tessedit_write_images=true'

    ocr_data = pytesseract.image_to_data(
        pre_processed_image,
        lang="slv",
        config=config,
        output_type=pytesseract.Output.DATAFRAME
    )

    # Drop rows where text is NaN or empty
    lines = []
    for line_num in ocr_data['line_num'].unique():
        line_text = ocr_data[ocr_data['line_num'] == line_num]['text']
        line_text = " ".join(line_text.dropna().astype(str))
        if line_text.strip():
            lines.append(line_text)

    raw_text_with_lines = "\n".join(lines)

    show_bounding_box(ocr_data, pre_processed_image)

    return raw_text_with_lines

    return post_process(ocr_data)


def perform_preprocess(img):

    image = img.copy()

    # Save initial image to debug
    cv2.imwrite("../results/01_initial.jpg", image)

    # Resize to enlarge small text
    scale_percent = 200  # 200% size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("../results/02_grey.jpg", image)

    # Invert colors (Tesseract works better with dark text on light background)
    # image = cv2.bitwise_not(image)
    cv2.imwrite("../results/03_inverted.jpg", image)

    # blur the image to reduce noise
    image = cv2.medianBlur(image, 5)
    cv2.imwrite("../results/04_blurred.jpg", image)

    # Apply adaptive thresholding (better for uneven lighting)
    image = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 5
    )
    cv2.imwrite("../results/05_thresh.jpg", image)

    return image


# def post_process(ocr_data):
#
#     ocr_data = ocr_data[ocr_data["text"].notnull() & (ocr_data["text"].str.strip() != "")]
#     ocr_data = ocr_data[ocr_data["conf"] > 30]  # Filter out low-confidence results
#     ocr_text = " ".join(ocr_data["text"].tolist())
#     return ocr_text


def show_bounding_box(ocr_data, img):
    image = img.copy()
    for i in range(len(ocr_data["text"])):
        text = str(ocr_data["text"][i])  # <-- make sure it's a string
        conf = int(ocr_data["conf"][i]) if ocr_data["conf"][i] != '' else -1

        if text.strip() != "" and text.lower() != "nan" and conf > 0:
            x, y, w, h = (
                ocr_data["left"][i],
                ocr_data["top"][i],
                ocr_data["width"][i],
                ocr_data["height"][i],
            )

            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Put text + confidence above box
            cv2.putText(
                image,
                f"{text} ({conf})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite("../results/06_bounding_boxes.jpg", image)


# from transformers import MBartForConditionalGeneration, MBart50TokenizerFast


# def post_process(ocr_data):
#     model_name = "facebook/mbart-large-50-many-to-many-mmt"
#     tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
#     model = MBartForConditionalGeneration.from_pretrained(model_name)
#
#     # Set target language to Slovenian
#     tokenizer.src_lang = "sl_SI"
#
#     # Load mBART model
#     ocr_data = ocr_data[ocr_data["text"].notnull() & (ocr_data["text"].str.strip() != "")]
#     #ocr_data = ocr_data[ocr_data["conf"] > 30]  # Filter out low-confidence results
#     noisy_text = " ".join(ocr_data["text"].tolist())
#     noisy_text = " Mlečna čokoladascelimi lešniki, Sestavine: sladkor, 27% lešniki, kakavovo maslo, polno .| »Giše Dlekov prahu, kakavova masa, sladka sirotka v prahu, laktoz_a, lešnikova pasta, emulgator. lecitini (sončnični lecitin); naravna aroma | Pl cvilje. V mlečni čokoladi kakavovi deli: najmanj 32%, Lahko vsebuje sledi drugih oreškov, zmja soje, arašidov in žit, ki vsebujejo | CŠL? S-dten, Uporabno najmanj do: glej odtis na zadnjistrani embalaže, Hranitivsuhem "
#
#     import re
#     noisy_text = re.sub(r"\s+", " ", noisy_text).strip()
#
#     # Tokenize input
#     inputs = tokenizer(noisy_text, return_tensors="pt")
#
#     # Generate output
#     translated_tokens = model.generate(
#         **inputs,
#         forced_bos_token_id=tokenizer.lang_code_to_id["sl_SI"],
#         max_length=512,
#         num_beams=5,
#         early_stopping=True
#     )
#
#     # Decode output
#     cleaned_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
#     print(cleaned_text)
#     return cleaned_text

def post_process(ocr_data):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
    model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")

    ocr_data = ocr_data[ocr_data["text"].notnull() & (ocr_data["text"].str.strip() != "")]
    ocr_data = ocr_data[ocr_data["conf"] > 30]  # Filter out low-confidence results
    ocr_text = " ".join(ocr_data["text"].tolist())

    words = ocr_text.split()
    sl_words = []

    for w in words:
        inputs = tokenizer(w, return_tensors="pt", truncation=True)
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1)
        lang = model.config.id2label[pred.item()]
        if lang == "sl":  # 'sl' is Slovenian
            sl_words.append(w)

    paragraph = ' '.join(sl_words)
    paragraph = paragraph.replace("\n", " ").replace("  ", " ")
    print(paragraph)
    print("helo")
