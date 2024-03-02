import numpy as np

def calculate_entropy(image):
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    histogram = np.histogram(image, bins=256, range=(0, 255))[0]
    probabilities = histogram / float(np.sum(histogram))
    entropy = -np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps))
    return entropy

if __name__ == "__main__":
    img = np.random.randint(0, 256, (100, 100, 3))
    print(calculate_entropy(img))