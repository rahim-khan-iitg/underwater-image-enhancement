import cv2
import os

def clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return enhanced_image

if __name__ == "__main__":
    input_dir = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/raw"
    output_dir = "D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/clahe"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files=os.listdir(input_dir)
    for file in files:
        image = cv2.imread(os.path.join(input_dir, file))
        enhanced_image = clahe(image)
        cv2.imwrite(os.path.join(output_dir, file), enhanced_image)
