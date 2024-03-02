import numpy as np
from skimage import filters,io

def calculate_uicm(image:np.ndarray,alpha_R,alpha_L)->float:
    R,G,B=image[:,:,0],image[:,:,1],image[:,:,2]
    RG=(R-G)
    YB=(0.5*(R+G)-B)
    rows,cols=image.shape[0],image.shape[1]
    K=rows*cols
    RG=RG.flatten()
    RG=np.sort(RG)
    YB=YB.flatten()
    YB=np.sort(YB)
    T_alpha_L=np.ceil(K*alpha_L).astype(int)
    T_alpha_R=np.floor(K*alpha_R).astype(int)
    sum_RG=np.sum(RG[T_alpha_L+1:K-T_alpha_R])
    mean_RG=sum_RG/(K-T_alpha_L-T_alpha_R)
    sum_YB=np.sum(YB[T_alpha_L+1:K-T_alpha_R])
    mean_YB=sum_YB/(K-T_alpha_L-T_alpha_R)
    var_RG=np.sum(np.power(RG[T_alpha_L+1:K-T_alpha_R]-mean_RG,2))/(K-T_alpha_L-T_alpha_R)
    var_YB=np.sum(np.power(YB[T_alpha_L+1:K-T_alpha_R]-mean_YB,2))/(K-T_alpha_L-T_alpha_R)
    UICM=-0.0268*np.sqrt(mean_RG**2 + mean_YB**2) + 0.1586*np.sqrt(var_RG + var_YB)
    return UICM

def calculate_uism(img:np.ndarray)->float:
    BLOCKSIZE = 8
    LAMBDA_R = 0.299
    LAMBDA_G = 0.587
    LAMBDA_B = 0.114

    b_channel, g_channel, r_channel = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    r_sobel = filters.sobel(r_channel)
    g_sobel = filters.sobel(g_channel)
    b_sobel = filters.sobel(b_channel)

    r_edge = (r_channel/255) * (r_sobel/255)*255
    g_edge = (g_channel/255) * (g_sobel/255)*255
    b_edge = (b_channel/255) * (b_sobel/255)*255

    sum_r, sum_g, sum_b = 0.0, 0.0, 0.0
    k1, k2 = img.shape[0] // BLOCKSIZE, img.shape[1] // BLOCKSIZE
    for i in range(k1):
        for j in range(k2):
            for edge, sum_val in zip([r_edge, g_edge, b_edge], [sum_r, sum_g, sum_b]):
                block = edge[i*BLOCKSIZE:(i+1)*BLOCKSIZE, j*BLOCKSIZE:(j+1)*BLOCKSIZE]
                min_val, max_val = np.min(block), np.max(block)
                if min_val != 0 and max_val != 0:
                    sum_val += np.log(max_val / min_val)
                else:
                    sum_val += 1

    sum_r *= 2.0 / (k1 * k2)
    sum_g *= 2.0 / (k1 * k2)
    sum_b *= 2.0 / (k1 * k2)

    result = LAMBDA_R * sum_r + LAMBDA_G * sum_g + LAMBDA_B * sum_b
    return result


PLIP_MU = 1026.0
PLIP_GAMMA = 1026.0
PLIP_K = 1026.0
PLIP_LAMBDA = 1026.0
PLIP_BETA = 1.0
BLOCKSIZE = 4

def calculate_uiconm(img):
    result = 0.0
    tempResult = 0.0

    k1 = img.shape[0] // BLOCKSIZE
    k2 = img.shape[1] // BLOCKSIZE

    for i in range(k1):
        for j in range(k2):
            min_val, max_val = findMinMaxIntensity(img, j*BLOCKSIZE, (j+1)*BLOCKSIZE, i*BLOCKSIZE, (i+1)*BLOCKSIZE)
            if min_val != max_val:
                subtraction = plipSubtraction(plipG(max_val), plipG(min_val))
                addition = plipAddition(plipG(max_val), plipG(min_val))
                tempResult = subtraction / addition

                result += plipMultiplication(plipG(tempResult), plipG(np.log(np.abs(tempResult))))

    c = 1 / (k1 * k2)

    result = plipScalarMultiplication(c, plipG(result))
    return result

def plipG(intensity):
    return (PLIP_MU - intensity)

def plipAddition(g1, g2):
    return (g1 + g2 - (g1 * g2) / PLIP_GAMMA)

def plipSubtraction(g1, g2):
    return (PLIP_K * (g1 - g2) / (PLIP_K - g2))

def plipScalarMultiplication(c, g):
    return (PLIP_GAMMA - PLIP_GAMMA * np.power(1 - g / PLIP_GAMMA, c))

def plipMultiplication(g1, g2):
    return plipPhiInverse(plipPhi(plipG(g1)) * plipPhi(plipG(g2)))

def plipPhi(g):
    return -PLIP_LAMBDA * np.power(np.log(1 - g / PLIP_LAMBDA), PLIP_BETA)

def plipPhiInverse(g):
    return PLIP_LAMBDA * (1 - np.power(np.exp(-g / PLIP_LAMBDA), 1 / PLIP_BETA))

def findMinMaxIntensity(img, xmin, xmax, ymin, ymax):
    block = img[ymin:ymax, xmin:xmax]
    intensity = np.mean(block, axis=2)
    return np.min(intensity), np.max(intensity)

def calculate_UIQM(img):
    alpha_R = 0.1
    alpha_L = 0.1
    c1=0.0282
    c2=0.2953
    c3=3.5753
    uiqm=c1*calculate_uicm(img,alpha_R,alpha_L)+c2*calculate_uism(img)+c3*calculate_uiconm(img)
    return uiqm


if __name__ == "__main__":
    img = io.imread('D:/MSc Books/Sem 4/Project/underwater_image_enhancement/images/d_r_1_.jpg')
    print(calculate_UIQM(img))