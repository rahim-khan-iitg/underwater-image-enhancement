import cv2
import numpy as np

def clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced_image

image_path = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/raw/test22.png"
img=cv2.imread(image_path)
enhanced_image = clahe(img)
cv2.imshow("Original Image", img)
cv2.imshow("Enhanced Image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
