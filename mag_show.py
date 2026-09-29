import cv2
import numpy as np
import matplotlib.pyplot as plt


def deskew(image):
    """
    Deskew a binary image using the minimum area rectangle method.
    """
    coords = np.column_stack(np.where(image > 0))

    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


# ----------------------------
# Load image
# ----------------------------
img = cv2.imread("dd.jpg")

if img is None:
    raise FileNotFoundError("Could not load image.")

# OpenCV loads BGR, convert for display
original_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ----------------------------
# 1. Grayscale
# ----------------------------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------------------------
# 2. Normalization
# Improves contrast
# ----------------------------
normalized = cv2.normalize(
    gray,
    None,
    alpha=0,
    beta=255,
    norm_type=cv2.NORM_MINMAX
)

# Alternative:
# normalized = cv2.equalizeHist(gray)

# ----------------------------
# 3. Thresholding
# Otsu automatically finds threshold
# ----------------------------
_, thresh = cv2.threshold(
    normalized,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Invert so text is white for deskew
thresh_inv = cv2.bitwise_not(thresh)

# ----------------------------
# 4. Deskew
# ----------------------------
deskewed = deskew(thresh_inv)

# ----------------------------
# Plot results
# ----------------------------
fig, axes = plt.subplots(1,4, figsize=(20, 5))

axes[0].imshow(original_rgb)
axes[0].set_title("Originalna slika")
axes[0].axis("off")

axes[1].imshow(gray, cmap="gray")
axes[1].set_title("Sivinska slika")
axes[1].axis("off")

axes[2].imshow(normalized, cmap="gray")
axes[2].set_title("Normalizacija")
axes[2].axis("off")

axes[3].imshow(thresh, cmap="gray")
axes[3].set_title("Binarizacija")
axes[3].axis("off")

#axes[4].imshow(deskewed, cmap="gray")
#axes[4].set_title("Skew Correction")
#axes[4].axis("off")

plt.tight_layout()
plt.show()