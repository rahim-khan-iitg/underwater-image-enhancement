import numpy as np
from skimage.color import rgb2lab
from skimage import io

def UCIQE(rgb_in:np.ndarray, c1:float=0.4680, c2:float=0.2745, c3:float=0.2576):
    lab:np.ndarray = rgb2lab(rgb_in)
    l:np.ndarray = lab[:,:,0]
    a:np.ndarray = lab[:,:,1]
    b:np.ndarray = lab[:,:,2]

    chroma:np.ndarray = np.sqrt(a**2 + b**2)
    u_c:float = np.mean(chroma)
    sigma_c:float = np.sqrt(np.mean(chroma**2 - u_c**2))

    saturation:np.ndarray = chroma / l
    u_s = np.mean(saturation)

    contrast_l:float = np.max(l) - np.min(l)

    UCIQE:float = c1 * sigma_c + c2 * contrast_l + c3 * u_s

    return UCIQE

if __name__ == "__main__":
    img=io.imread('D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/d_r_1_.jpg')

    print(UCIQE(img))