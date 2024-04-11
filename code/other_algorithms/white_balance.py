import numpy as np
from skimage import color
import cv2
import os

def whiteBalance(im):
    # Using Grayworld assumption color balancing.....

    R_avg = np.mean(im[:,:,0])  # Getting the average of R, G, B components
    G_avg = np.mean(im[:,:,1])
    B_avg = np.mean(im[:,:,2])
    RGB_avg = [R_avg, G_avg, B_avg]

    gray_value = (R_avg + G_avg + B_avg) / 3  # By Grey world, average color of the whole image is gray
    scaleValue = gray_value / RGB_avg  # By Grey world, scale value = gray / average of each color component

    whiteBalanced = np.zeros_like(im)  # Create an empty array for the white balanced new image
    whiteBalanced[:,:,0] = scaleValue[0] * im[:,:,0]  # R, G, B components of the new white balanced image
    whiteBalanced[:,:,1] = scaleValue[1] * im[:,:,1]
    whiteBalanced[:,:,2] = scaleValue[2] * im[:,:,2]

    return whiteBalanced

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

if __name__ == "__main__":
    input_dir = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/raw"
    output_dir = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/white_balance"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files=os.listdir(input_dir)
    for file in files:
        image = cv2.imread(os.path.join(input_dir, file))
        enhanced_image = whiteBalance(image)
        cv2.imwrite(os.path.join(output_dir, file), enhanced_image)