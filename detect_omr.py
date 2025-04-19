import cv2
import cv2.aruco as aruco
import numpy as np
from imutils.perspective import four_point_transform

# Read image
image = cv2.imread("assets/blank_omr.png")
if image is None:
    print("Error: Could not read the image file")
    exit(1)
    
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
detector = aruco.ArucoDetector(aruco_dict)
corners, ids, rejected = detector.detectMarkers(gray)

if ids is not None and len(ids) == 4:
    # Print detected IDs for debugging
    print("Detected marker IDs:", [id[0] for id in ids])
    
    # Create a mapping of ID to corners (taking the first corner point of each marker)
    id_corner_map = {}
    for marker_id, marker_corners in zip(ids, corners):
        # Each marker has 4 corners, we take the first one as reference
        id_corner_map[marker_id[0]] = marker_corners[0][0]
    
    # Sort corners in TL, TR, BR, BL order using IDs
    ordered_ids = [1, 2, 4, 3]  # Adjust based on where you placed each ID
    try:
        # Create array of corner points in the correct order
        ordered_pts = np.array([id_corner_map[i] for i in ordered_ids], dtype=np.float32)
        
        # Draw detected markers for debugging
        debug_img = image.copy()
        for marker_id, marker_corners in zip(ids, corners):
            cv2.polylines(debug_img, [marker_corners[0].astype(np.int32)], True, (0, 255, 0), 2)
            # Add marker ID text
            center = marker_corners[0].mean(axis=0).astype(np.int32)
            cv2.putText(debug_img, str(marker_id[0]), tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imwrite("output/detected_markers.jpg", debug_img)
        print("Saved debug image with detected markers")
        
        # Warp
        warped = four_point_transform(image, ordered_pts)
        cv2.imwrite("output/warped_omr.jpg", warped)
        print("Successfully warped and saved the image!")
        
    except KeyError as e:
        print(f"Error: Missing marker with ID {e}. Make sure all required markers (1,2,3,4) are visible.")
    except Exception as e:
        print(f"Error during warping: {e}")
else:
    print("All 4 markers not detected! Found markers:", len(ids) if ids is not None else 0)
