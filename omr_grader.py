import cv2
import numpy as np
import json
import os

# === Configuration ===
IMAGE_PATH = "output/detect_omr/warped_omr.jpg"
JSON_PATH = "assets/omr_coordinates.json"
FILL_THRESHOLD = 0.6  # Increased threshold for actual filled circles
CONFIDENCE_THRESHOLD = 0.4  # Increased confidence threshold
MIN_FILLED_AREA = 0.3  # Minimum area that must be filled
DEBUG_MODE = True  # Set to True to save debug images
DIGIT_HEIGHT_GAP = 20  # Pixel gap between digits stacked vertically

# Create debug directory if needed
if DEBUG_MODE:
    os.makedirs('output/omr_grader', exist_ok=True)

# === Enhanced Image Preprocessing ===
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Remove noise
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    if DEBUG_MODE:
        cv2.imwrite('output/debug/preprocessed.jpg', thresh)
    
    return thresh

# === Enhanced Bubble Detection ===
def is_filled(img, x, y, r, threshold=FILL_THRESHOLD):
    # Use a smaller ROI to avoid text
    inner_r = int(r * 0.7)  # Use 70% of the radius to avoid text
    roi = img[y - inner_r:y + inner_r, x - inner_r:x + inner_r]
    
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return False, 0.0
    
    # Create circular mask for the inner region
    mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.circle(mask, (inner_r, inner_r), inner_r, 255, -1)
    
    # Calculate filled ratio
    total_pixels = np.sum(mask == 255)
    filled_pixels = np.sum((roi == 255) & (mask == 255))
    fill_ratio = filled_pixels / total_pixels
    
    # Calculate confidence score
    confidence = min(1.0, fill_ratio / threshold)
    
    # Check if the filled area is significant enough
    is_significant_fill = fill_ratio > threshold and (filled_pixels / (np.pi * r * r)) > MIN_FILLED_AREA
    
    return is_significant_fill, confidence

# === Load and Process Image ===
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"Could not read image at {IMAGE_PATH}")

thresh = preprocess_image(img)

# === Load bubble coordinates ===
with open(JSON_PATH, "r") as f:
    data = json.load(f)

results = {
    "student_id": "",
    "exam_date": "",
    "answers": {},
    "confidence_scores": {}
}

# === Detect student ID digits ===
for bubble in data.get("student_id", []):
    max_confidence = 0
    selected_digit = "X"
    
    for digit in range(10):
        x = bubble["x"]
        y = bubble["y"] + digit * DIGIT_HEIGHT_GAP
        r = bubble["r"]
        is_filled_bubble, confidence = is_filled(thresh, x, y, r)
        
        if is_filled_bubble and confidence > max_confidence:
            max_confidence = confidence
            selected_digit = str(digit)
    
    results["student_id"] += selected_digit
    results["confidence_scores"]["student_id"] = max_confidence

# === Detect exam date digits ===
for bubble in data.get("exam_date", []):
    max_confidence = 0
    selected_digit = "X"
    
    for digit in range(10):
        x = bubble["x"]
        y = bubble["y"] + digit * DIGIT_HEIGHT_GAP
        r = bubble["r"]
        is_filled_bubble, confidence = is_filled(thresh, x, y, r)
        
        if is_filled_bubble and confidence > max_confidence:
            max_confidence = confidence
            selected_digit = str(digit)
    
    results["exam_date"] += selected_digit
    results["confidence_scores"]["exam_date"] = max_confidence

# === Detect MCQ answers ===
for q_key, q_bubbles in data.get("questions", {}).items():
    max_confidence = 0
    selected_answer = "Not marked"
    multiple_marks = False
    
    for i, bubble in enumerate(q_bubbles):
        x, y, r = bubble["x"], bubble["y"], bubble["r"]
        is_filled_bubble, confidence = is_filled(thresh, x, y, r)
        
        if is_filled_bubble:
            if confidence > max_confidence:
                max_confidence = confidence
                selected_answer = chr(65 + i)  # A, B, C, D
            else:
                multiple_marks = True
    
    if max_confidence < CONFIDENCE_THRESHOLD:
        selected_answer = "Not marked"
    elif multiple_marks:
        selected_answer = "Multiple marks detected"
    
    results["answers"][q_key] = selected_answer
    results["confidence_scores"][q_key] = max_confidence

# === Output result ===
print(json.dumps(results, indent=2))

# === Save debug visualization if enabled ===
if DEBUG_MODE:
    debug_img = img.copy()
    for q_key, q_bubbles in data.get("questions", {}).items():
        for i, bubble in enumerate(q_bubbles):
            x, y, r = bubble["x"], bubble["y"], bubble["r"]
            # Draw outer circle
            cv2.circle(debug_img, (x, y), r, (0, 0, 255), 2)
            # Draw inner circle (detection area)
            inner_r = int(r * 0.7)
            cv2.circle(debug_img, (x, y), inner_r, (255, 0, 0), 1)
            
            # Color based on detection
            color = (0, 255, 0) if results["answers"][q_key] == chr(65 + i) else (0, 0, 255)
            cv2.putText(debug_img, f"{q_key}_{chr(65 + i)}", (x - r, y - r - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite('output/debug/detected_bubbles.jpg', debug_img)
