import numpy as np

from src.phenotype import phenotype, phenotype_opaque
from src.fitness import fitness_ssim, fitness_mse, prep_img


def evaluate(targetimage: np.ndarray, phenotypes: np.ndarray, evaluations: np.ndarray, mask: np.ndarray | None=None, resize_scale: float=1, method: str="ssim"):
    targetimage_scaled = prep_img(targetimage,resize_scale)
    if mask is None: mask = np.ones(phenotypes.shape[0])

    if method=="ssim":      fitness = fitness_ssim
    elif method=="mse":     fitness = fitness_mse
    else: raise ValueError(f"Method must be a valid method. Recieved \"{method}\"")

    best_i = 0; best_eval = -np.inf; best_img = None; best_img_heatmap = None

    for i in np.where(mask)[0]:
        img_scaled = prep_img(phenotypes[i], scale=resize_scale)
        s, _ = fitness(img_scaled, targetimage_scaled)
        evaluations[i] = s

    order = np.argsort(evaluations)[::-1]
    rankings = np.empty_like(order); rankings[order] = np.arange(len(order))

    best_i = order[0]
    best_img = prep_img(phenotypes[best_i],scale=resize_scale)
    best_eval, best_img_heatmap = fitness(best_img,   targetimage_scaled)

    return rankings, (best_i,best_eval,best_img,best_img_heatmap)

"""
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
"""