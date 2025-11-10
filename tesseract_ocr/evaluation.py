from rapidfuzz.distance import Levenshtein


def cer(ocr_text, ground_truth):
    """
    Calculate the Character Error Rate (CER) between OCR text and ground truth.
    CER is defined as the Levenshtein distance divided by the length of the ground truth.
    """
    if len(ground_truth) == 0:
        return float('inf') if len(ocr_text) > 0 else 0.0

    distance = Levenshtein.distance(ocr_text, ground_truth)
    return distance / len(ground_truth)


def wer(ocr_text, ground_truth):
    """
    Calculate the Word Error Rate (WER) between OCR text and ground truth.
    WER is defined as the Levenshtein distance between the word lists divided by the number of words in the ground truth.
    """
    ocr_words = ocr_text.split()
    gt_words = ground_truth.split()

    if len(gt_words) == 0:
        return float('inf') if len(ocr_words) > 0 else 0.0

    distance = Levenshtein.distance(ocr_words, gt_words)
    return distance / len(gt_words)
