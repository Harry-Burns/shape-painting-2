import numpy as np

from src.phenotype import phenotype, phenotype_opaque
from src.fitness import fitness_ssim, fitness_mse, prep_img, fitness_hybrid


def evaluate(targetimage: np.ndarray, population: np.ndarray, skip_mask=np.ndarray, detail: float=1, resize_scale: float=1, method: str="ssim") -> np.ndarray:
    h,w = targetimage.shape[:2]

    targetimage_scaled = prep_img(targetimage,resize_scale)

    evaluations = np.empty(population.shape[0], dtype=float)
    best_i = 0; best_eval = -np.inf; best_img = None; best_img_heatmap = None

    for i, g in enumerate(population):
        img = phenotype_opaque(genotype=g, image_data=(w,h), detail=detail)
        img_scaled = prep_img(img,scale=resize_scale)

        if method=="ssim":
            s, hm = fitness_ssim(img_scaled,targetimage_scaled)
        elif method=="mse":
            s, hm = fitness_mse(img,targetimage_scaled)
        elif method=="hybrid":
            s, hm = fitness_hybrid(g,img,targetimage_scaled)
        evaluations[i] = s
        if s > best_eval:
            best_eval = s; best_i = i; best_img = img; best_img_heatmap = hm

    order = np.argsort(evaluations)[::-1]
    rankings = np.empty_like(order); rankings[order] = np.arange(len(order))

    return evaluations, rankings, (best_i,best_eval,best_img,best_img_heatmap)


def evaluate_iter(targetimage: np.ndarray, population: np.ndarray, elite_idx: np.ndarray = None, detail: float=1, iter_info: dict = None) -> np.ndarray:
    if iter_info is None: print("No iter_info given, using default evaluation method."); return evaluate(targetimage,population)

    N = population.shape[0]

    evals_full = np.zeros(N,dtype=float)
    ranks_full = np.full(N, -1, dtype=np.int32)

    all_idx = np.arange(N)

    if elite_idx is not None and elite_idx.size:
        mask = np.ones(N, dtype=bool)
        mask[elite_idx] = False
        remaining_idx = all_idx[mask]
    else:
        remaining_idx = all_idx

    for i,(quality,keep,method) in enumerate(zip(iter_info['quality'],iter_info['keep'],iter_info['method'])):
        keep_n = min(max(2, int(np.ceil(N * keep))), len(remaining_idx))
        if i == len(iter_info["keep"]) - 1 or keep_n == len(remaining_idx): break

        evals_sub, ranks_sub, best_data = evaluate(targetimage, population[remaining_idx], detail, resize_scale=quality, method=method)

        evals_full[remaining_idx] = evals_sub
        ranks_full[remaining_idx] = ranks_sub

        sub_keep = np.argpartition(ranks_sub, keep_n - 1)[:keep_n]
        remaining_idx = remaining_idx[sub_keep]

    final_quality = iter_info["quality"][-1]; final_method  = iter_info["method"][-1] 
    final_idx = np.concatenate([remaining_idx, elite_idx]) if elite_idx is not None and elite_idx.size else remaining_idx

    evals_sub, ranks_sub, best_data = evaluate(targetimage, population[final_idx], detail, resize_scale=final_quality, method=final_method)

    evals_full[final_idx] = evals_sub
    ranks_full[final_idx] = ranks_sub

    return evals_full, ranks_full, best_data