import cv2
import numpy as np
import json

# === Load OMR Image ===
image_path = "output/detect_omr/warped_omr.jpg"
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# === Preprocessing ===
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# === Load Bubble Coordinates ===
with open("assets/omr_coordinates.json", "r") as f:
    coords = json.load(f)

# === Answer Detection ===
answers = {}

# Draw copy for debugging
debug_image = image.copy()

def get_mean_intensity(x, y, r, binary_img):
    mask = np.zeros_like(binary_img, dtype=np.uint8)
    cv2.circle(mask, (x, y), r, 255, -1)
    masked = cv2.bitwise_and(binary_img, binary_img, mask=mask)
    mean_val = cv2.mean(masked, mask=mask)[0]
    return mean_val

for q_key, bubbles in coords["questions"].items():
    intensities = []
    for i, bubble in enumerate(bubbles):
        x, y, r = bubble["x"], bubble["y"], bubble["r"]
        mean_val = get_mean_intensity(x, y, r, thresh)
        intensities.append((i, mean_val))

        # Draw circle for debugging
        cv2.circle(debug_image, (x, y), r, (0, 255, 0), 2)
        cv2.putText(debug_image, chr(ord("A") + i), (x - 10, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Choose lowest mean (darkest)
    marked_index = min(intensities, key=lambda x: x[1])[0]
    answers[q_key] = chr(ord("A") + marked_index)

    # Highlight selected answer
    selected = bubbles[marked_index]
    cv2.circle(debug_image, (selected["x"], selected["y"]), selected["r"], (0, 0, 255), 2)

# === Output ===
print("✅ Detected Answers:")
print(json.dumps(answers, indent=2))

# === Save Debug Images ===
cv2.imwrite("output_debug_omr.jpg", debug_image)
cv2.imwrite("step1_gray.jpg", gray)
cv2.imwrite("step2_thresh.jpg", thresh)
