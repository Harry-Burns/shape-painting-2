import numpy as np
import cv2

def fitness_hybrid(genes,img,target):
    s, hm = fitness_ssim(img,target)
    alpha_mean = np.mean(genes[:,5]) # 5 is alpha
    return float(s + alpha_mean), hm

def fitness_ssim(img, target):
    ssim, hmap = cv2.quality.QualitySSIM_compute(img, target)
    return float(ssim[0]), hmap  # if grayscale; if colour you may want mean()

def fitness_mse(img, target): 
    diff = img.astype(np.float32) - target.astype(np.float32)
    hmap = diff * diff

    mse = float(np.mean(hmap))

    return -mse, hmap

def prep_img(img,scale: float=1):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if scale != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img
