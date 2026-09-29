import cv2
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Create canvas
# ----------------------------
canvas_size = 400
canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255

# Draw character
font = cv2.FONT_HERSHEY_SIMPLEX
text = "h"

(text_w, text_h), baseline = cv2.getTextSize(
    text,
    font,
    8,
    15
)

x = (canvas_size - text_w) // 2
y = (canvas_size + text_h) // 2

cv2.putText(
    canvas,
    text,
    (x, y),
    font,
    8,
    (0,),
    15,
    cv2.LINE_AA
)

# ----------------------------
# Grid settings
# ----------------------------
rows = 10
cols = 10

cell_h = canvas_size // rows
cell_w = canvas_size // cols

# Grid visualization image
grid_img = cv2.cvtColor(canvas.copy(), cv2.COLOR_GRAY2RGB)

# Occupancy visualization image
occupancy_img = np.ones_like(grid_img) * 255

# ----------------------------
# Draw grid
# ----------------------------
for r in range(rows + 1):
    y_line = r * cell_h
    cv2.line(
        grid_img,
        (0, y_line),
        (canvas_size, y_line),
        (255, 0, 0),
        1
    )

for c in range(cols + 1):
    x_line = c * cell_w
    cv2.line(
        grid_img,
        (x_line, 0),
        (x_line, canvas_size),
        (255, 0, 0),
        1
    )

# ----------------------------
# Check occupancy
# ----------------------------
for r in range(rows):
    for c in range(cols):

        y1 = r * cell_h
        y2 = (r + 1) * cell_h

        x1 = c * cell_w
        x2 = (c + 1) * cell_w

        cell = canvas[y1:y2, x1:x2]

        # Any dark pixel?
        occupied = np.any(cell < 128)

        cx = x1 + cell_w // 2
        cy = y1 + cell_h // 2

        symbol = "1" if occupied else "0"

        cv2.putText(
            occupancy_img,
            symbol,
            (cx - 12, cy + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

# ----------------------------
# Plot
# ----------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

#axes[0].imshow(canvas, cmap="gray")
##axes[0].set_title("")
#axes[0].axis("off")

axes[0].imshow(grid_img)
axes[0].set_title("")
axes[0].axis("off")

axes[1].imshow(occupancy_img)
axes[1].set_title("")
axes[1].axis("off")

plt.tight_layout()
plt.show()