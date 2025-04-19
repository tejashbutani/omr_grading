import cv2
import numpy as np
import json

# === Configuration ===
IMAGE_PATH = "output/detect_omr/warped_omr.jpg"
JSON_PATH = "assets/omr_coordinates.json"
FILL_THRESHOLD = 0.5  # Tune based on marker darkness
DIGIT_HEIGHT_GAP = 20  # Pixel gap between digits stacked vertically

# === Load image and preprocess ===
img = cv2.imread(IMAGE_PATH)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# === Function to check if bubble is filled ===
def is_filled(img, x, y, r, threshold=FILL_THRESHOLD):
    roi = img[y - r:y + r, x - r:x + r]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return False
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(mask, (r, r), r, 255, -1)
    total_pixels = np.sum(mask == 255)
    filled_pixels = np.sum((roi == 255) & (mask == 255))
    return (filled_pixels / total_pixels) > threshold

# === Load bubble coordinates ===
with open(JSON_PATH, "r") as f:
    data = json.load(f)

results = {
    "student_id": "",
    "exam_date": "",
    "answers": {}
}

# === Detect student ID digits ===
for bubble in data.get("student_id", []):
    for digit in range(10):
        x = bubble["x"]
        y = bubble["y"] + digit * DIGIT_HEIGHT_GAP
        r = bubble["r"]
        if is_filled(thresh, x, y, r):
            results["student_id"] += str(digit)
            break
    else:
        results["student_id"] += "X"  # unmarked fallback

# === Detect exam date digits ===
for bubble in data.get("exam_date", []):
    for digit in range(10):
        x = bubble["x"]
        y = bubble["y"] + digit * DIGIT_HEIGHT_GAP
        r = bubble["r"]
        if is_filled(thresh, x, y, r):
            results["exam_date"] += str(digit)
            break
    else:
        results["exam_date"] += "X"

# === Detect MCQ answers ===
for q_key, q_bubbles in data.get("questions", {}).items():
    marked = False
    for i, bubble in enumerate(q_bubbles):
        x, y, r = bubble["x"], bubble["y"], bubble["r"]
        if is_filled(thresh, x, y, r):
            results["answers"][q_key] = chr(65 + i)  # A, B, C, D
            marked = True
            break
    if not marked:
        results["answers"][q_key] = "Not marked"

# === Output result ===
print(json.dumps(results, indent=2))
