import numpy as np
import cv2
import os
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

if __name__ == "__main__":
    input_dir = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/raw"
    output_dir = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/max_rgb"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files=os.listdir(input_dir)
    for file in files:
        image = cv2.imread(os.path.join(input_dir, file))
        enhanced_image = max_rgb(image)
        cv2.imwrite(os.path.join(output_dir, file), enhanced_image)
