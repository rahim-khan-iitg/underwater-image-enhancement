import numpy as np
from skimage import feature,color

def CCFcontrast(img:np.ndarray)->float:
    img=color.rgb2gray(img)
    T = 0.002
    img = img.astype(float)
    m, n = img.shape
    rb = 64
    rc = 64
    count = 1
    contrast_JNB = []

    for i in range(m//rb):
        for j in range(n//rc):
            row = slice(rb*(i), rb*(i+1))
            col = slice(rc*(j), rc*(j+1))
            img_temp = img[row, col]
            decision = get_edgeblocks_mod(img_temp, T)
            if decision:
                contrast_JNB.append(get_contrast_block(img_temp))
                count += 1

    L = (m//rb) * (n//rc)
    metric = sum(contrast_JNB) / L
    return metric

def get_edgeblocks_mod(im_in, T):
    im_in = im_in.astype(float)
    im_in_edge = feature.canny(im_in)
    L = np.size(im_in_edge)
    im_edge_pixels = np.sum(im_in_edge)
    return im_edge_pixels > (L * T)

def get_contrast_block(img:np.ndarray)->float:
    m, n = img.shape
    contrast_wy = np.sqrt(np.sum((img - np.mean(img))**2) / (n * m))
    return contrast_wy


def ccf(img:np.ndarray)->float:
    R,G,B=img[:,:,0],img[:,:,1],img[:,:,2]
    RR=np.log10(R+0.00001)-np.mean(np.log10(R+0.00001))
    GG=np.log10(G+0.00001)-np.mean(np.log10(G+0.00001))
    BB=np.log10(B+0.00001)-np.mean(np.log10(B+0.00001))
    alpha=RR-GG
    beta=0.5*(RR+GG)-BB
    mu_alpha=np.mean(alpha)
    mu_beta=np.mean(beta)
    var_alpha=np.var(alpha)
    var_beta=np.var(beta)
    CCF_colorfulness=1000*((np.sqrt(var_alpha+var_beta)+0.3*np.sqrt(mu_alpha*mu_alpha+mu_beta*mu_beta))/85.59)

    return CCF_colorfulness


if __name__ == "__main__":
    img = np.random.rand(100,100,3)
    print(CCFcontrast(img))