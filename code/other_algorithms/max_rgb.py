import numpy as np
import cv2
def max_rgb(image):
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    max_values = np.maximum(np.maximum(red, green), blue)
    enhanced_image = np.zeros_like(image)
    enhanced_image[:, :, 0] = max_values
    enhanced_image[:, :, 1] = max_values
    enhanced_image[:, :, 2] = max_values

    return enhanced_image

image_path = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/raw/test2.png"
image = cv2.imread(image_path)
enhanced_image = max_rgb(image)
cv2.imshow("Original Image", image)
cv2.imshow("Enhanced Image", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
