import os
import cv2


def load_images(folder_path="../images/sestavine"):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img_id = filename.split("_")[0]
            if img is not None:
                images.append((img_id, img))
    return images


def load_annotations(file_path="../annotations.txt"):
    annotations = {}
    with open(file_path, "r", encoding="utf-8") as file:
        id = file.readline().strip()
        content = file.readline().strip()
        annotations[id] = content
    return annotations


def load_data(image_folder="../images/sestavine", annotation_file="../annotations.txt"):
    data = []
    images = load_images(image_folder)
    annotations = load_annotations(annotation_file)

    for img_id, img in images:
        if img_id in annotations:
            data.append((annotations[img_id], img))  # tuple: (annotation, image)
    return data

