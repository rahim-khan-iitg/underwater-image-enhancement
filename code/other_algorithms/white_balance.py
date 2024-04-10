import numpy as np
from skimage import color
import cv2

def white_balance(image):
    lab = color.rgb2lab(image)
    l, a, b = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    mean_a, std_a = np.mean(a), np.std(a)
    mean_b, std_b = np.mean(b), np.std(b)
    a = np.clip(a - mean_a * (std_b / std_a), 0, 100)
    b = np.clip(b - mean_b * (std_a / std_b), 0, 100)
    lab[:,:,1], lab[:,:,2] = a, b
    balanced_image = color.lab2rgb(lab)

    return balanced_image

image_path = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/raw/test22.png"
image = cv2.imread(image_path)
enhanced_image = white_balance(image)
cv2.imshow("Original Image", image)
cv2.imshow("Enhanced Image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

