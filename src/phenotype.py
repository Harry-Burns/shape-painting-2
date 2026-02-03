from src.genotype import Genotype

import numpy as np
import cv2

cv2.ocl.setUseOpenCL(False) 

# Species consists of a list of "GENES"
#   1. There is a maximum number of genes
#   2. Each gene consists of 3 pieces of information, each of which is stored from 0-1
#       2.1. Size, w/h ratio, position (x, y), color (greyscale/rgb + opacity), index drawing order
#       2.2. This means each gene (row) in the np array will be 1+1+2+2+1 long


def phenotype(genotype: Genotype | np.ndarray, image_data: tuple[int,int], detail: float=1) -> np.ndarray:
    if isinstance(genotype, Genotype): genes = genotype.genes 
    else: genes = genotype

    width,height = image_data
    min_size = int(min(width,height)/100)
    max_size = int(max(width,height)/2)

    num_genes = Genotype.max_genes

    # bounds
    size_b = (min_size,max_size)    # col 0
    ratio_b = (0.05,0.95)           # col 1
    posx_b = (0,width)              # col 2
    posy_b = (0,height)             # col 3
    col_b = (0,255)                 # col 4
    alpha_b = (0,1)                 # col 5
    idx_b = (0, num_genes - 1)      # col 6

    bounds = np.array([size_b, ratio_b, posx_b, posy_b, col_b, alpha_b, idx_b], dtype=float)
    mappings = [map_log, map_linear, map_linear, map_linear, map_linear, map_linear, map_linear]

    circle_data = np.empty_like(genes, dtype=float)
    for j, f in enumerate(mappings):
        low, high = bounds[j]
        circle_data[:, j] = f(genes[:, j], low, high)

    circle_data_s = circle_data[np.argsort(genes[:, 6])]    

    # image generation
    img = np.full((height, width), 128, dtype=np.uint8)
    mask_buf = np.zeros((height, width), dtype=np.uint8)

    n_draw = int(max(min(detail,1),0) * num_genes)

    for circle in circle_data_s[-n_draw:]:
        radius, ratio, posx, posy, col1, alpha, _ = circle

        # example colour mapping (you can change this)
        #color = (
        #    int(col1),
        #    int(col1),
        #    int(col1)
        #)
        color = int(col1)

        draw_circle(img,mask_buf,radius,ratio,posx,posy,color,alpha)
    
    return np.asarray(img)

def draw_circle(img_bgr, mask_buf, radius, ratio, posx, posy, col_bgr, alpha):
    min_alpha = 0.01; min_rad = 1
    if alpha <= min_alpha or radius <= min_rad or ratio == 0 or ratio == 1:  return

    h,w = img_bgr.shape[:2]

    x0 = max(int(posx - radius * ratio), 0); x1 = min(int(posx + radius * ratio + 1), w)
    y0 = max(int(posy - radius * (1-ratio)), 0); y1 = min(int(posy + radius * (1-ratio) + 1), h)
    if x0 >= x1 or y0 >= y1: return

    region_of_interest = img_bgr[y0:y1,x0:x1]

    # 1-channel mask for the circle in ROI coords
    mask = mask_buf[y0:y1, x0:x1]; mask.fill(0)
    cv2.ellipse(
        mask, center=(int(posx - x0), int(posy - y0)),
        axes=(int(radius * ratio), int(radius * (1-ratio))),
        angle=0, startAngle=0, endAngle=360, color=255, thickness=-1, lineType=cv2.LINE_AA,
    )

    # Blend only where mask > 0
    a = mask.astype(np.float32) * (float(alpha) / 255.0)

    # Do blend in float, write back to uint8   - THIS IS FOR COLOURED
    #src = np.asarray(col_bgr, dtype=np.float32)[None, None, :]
    #dst = region_of_interest.astype(np.float32)
#
    #out = dst * (1.0 - a[..., None]) + src * a[..., None]
    #region_of_interest[:] = np.clip(out, 0, 255).astype(np.uint8)

    # GREYSCALE
    src = float(col_bgr)                  # scalar intensity
    dst = region_of_interest.astype(np.float32)

    out = dst * (1.0 - a) + src * a        # a is (H,W)
    region_of_interest[:] = np.clip(out, 0, 255).astype(np.uint8)



# region Opaque Experiment
def phenotype_opaque(genotype: Genotype | np.ndarray, image_data: tuple[int,int], detail: float=1) -> np.ndarray:
    if isinstance(genotype, Genotype): genes = genotype.genes 
    else: genes = genotype

    width,height = image_data
    min_size = int(min(width,height)/100)
    max_size = int(max(width,height))

    num_genes = Genotype.max_genes

    # bounds
    size_b = (min_size,max_size)    # col 0
    ratio_b = (0.02,0.98)             # col 1
    posx_b = (0,width)              # col 2
    posy_b = (0,height)             # col 3
    col_b = (0,255)                 # col 4
    alpha_b = (0,1)                 # col 5
    idx_b = (0, num_genes - 1)      # col 6

    bounds = np.array([size_b, ratio_b, posx_b, posy_b, col_b, alpha_b, idx_b], dtype=float)
    mappings = [map_log, map_linear, map_linear, map_linear, map_linear, map_linear, map_linear]

    circle_data = np.empty_like(genes, dtype=float)
    for j, f in enumerate(mappings):
        low, high = bounds[j]
        circle_data[:, j] = f(genes[:, j], low, high)

    circle_data_s = circle_data[np.argsort(genes[:, 6])]    

    # image generation
    img = np.full((height, width), 128, dtype=np.uint8) # 

    n_draw = int(max(min(detail,1),0) * num_genes)

    for circle in circle_data_s[-n_draw:]:
        radius, ratio, posx, posy, col1, alpha, _ = circle

        # example colour mapping (you can change this)
        #color = (
        #    int(col1),
        #    int(col1),
        #    int(col1)
        #)
        color = int(col1)

        draw_circle_opaque(img,radius,ratio,posx,posy,color)
    
    return np.asarray(img)

def draw_circle_opaque(img, radius, ratio, posx, posy, col):
    h, w = img.shape[:2]

    ax = int(radius * ratio)
    ay = int(radius * (1 - ratio))
    if ax <= 0 or ay <= 0:
        return

    cx = int(posx)
    cy = int(posy)
    if cx + ax < 0 or cx - ax >= w or cy + ay < 0 or cy - ay >= h:
        return

    cv2.ellipse(
        img,
        center=(cx, cy),
        axes=(ax, ay),
        angle=0, startAngle=0, endAngle=360,
        color=int(col),
        thickness=-1,
        lineType=cv2.LINE_8,   # fast
    )
# endregion



def map_linear(gene: np.ndarray, low: float, high: float) -> np.ndarray:
    return low + gene * (high - low)

def map_log(gene: np.ndarray, low: float, high: float, eps: float = 1e-6) -> np.ndarray:
    low = max(low, eps)
    high = max(high, low + eps)
    return low * ((high / low) ** gene)