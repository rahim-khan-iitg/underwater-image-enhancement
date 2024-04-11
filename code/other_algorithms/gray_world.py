import cv2
import numpy as np
import os

def gray_world(image):
    # Convert the image to float32 for accurate calculations
    image = image.astype(np.float32)

    # Calculate the average values for each channel
    avg_r = np.mean(image[:, :, 0])
    avg_g = np.mean(image[:, :, 1])
    avg_b = np.mean(image[:, :, 2])

    # Calculate the average gray value
    avg_gray = (avg_r + avg_g + avg_b) / 3

    # Scale the image channels by the average gray value
    image[:, :, 0] = np.clip(image[:, :, 0] * (avg_gray / avg_r), 0, 255)
    image[:, :, 1] = np.clip(image[:, :, 1] * (avg_gray / avg_g), 0, 255)
    image[:, :, 2] = np.clip(image[:, :, 2] * (avg_gray / avg_b), 0, 255)

    # Convert the image back to uint8
    image = image.astype(np.uint8)

    return image

if __name__ == "__main__":
    input_dir = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/raw"
    output_dir = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/grey_world"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files=os.listdir(input_dir)
    for file in files:
        image = cv2.imread(os.path.join(input_dir, file))
        enhanced_image =gray_world (image)
        cv2.imwrite(os.path.join(output_dir, file), enhanced_image)
