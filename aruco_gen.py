import cv2
import cv2.aruco as aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)

for id in range(1, 5):
    marker = aruco.drawMarker(aruco_dict, id, 200)  # 200x200 px
    cv2.imwrite(f"marker_{id}.png", marker)