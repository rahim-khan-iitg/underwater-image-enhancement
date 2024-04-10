import cv2
import numpy as np

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

# Load the image
image_path = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/raw/test14.png"
image = cv2.imread(image_path)

# Apply the Gray-World algorithm
enhanced_image = gray_world(image)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', enhanced_image)
cv2.imwrite("D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/gray_world_test14.png", enhanced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()