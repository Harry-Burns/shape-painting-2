import time
import numpy as np
import cv2

from src.genotype import Genotype
from src.evaluate import evaluate_iter
from src.selection import select_and_mutate
from src.display import display_n_with_stats

POPULATION_SIZE = 1024
MAX_ITER = 64000

EVOLVE_ITER = {
    'quality': [0.4, 1],
    'keep': [0.4, 1],
    'method': ["ssim", "ssim"]
}

#EVOLVE_ITER = {
#    'quality': [1],
#    'keep': [1],
#    'method': ["ssim"] 
#}

def evolve(targetimage: np.ndarray, checkpoint: np.ndarray | None = None) -> np.ndarray:
    assert len(EVOLVE_ITER['quality']) == len(EVOLVE_ITER['keep']) == len(EVOLVE_ITER['method'])

    pop_shape = (POPULATION_SIZE, Genotype.gene_shape[0], Genotype.gene_shape[1])

    init_population = np.random.rand(pop_shape[0], pop_shape[1], pop_shape[2]);

    if checkpoint is None: pass; #init_population[:,:,5:6] = 0 # optional init params
    elif checkpoint.shape[1:] != pop_shape[1:]: raise ValueError(f"evolve: checkpoint size was wrong. Expected {pop_shape} got {checkpoint.shape}")
    elif checkpoint.shape[0] != POPULATION_SIZE:
        if checkpoint.shape[0] < POPULATION_SIZE:
            print("Checkpoint population size too small, adding random population...")
            init_population[:checkpoint.shape[0],:,:] = checkpoint[:,:,:]
        else:
            print("Checkpoint population size too large.")
            init_population[:,:,:] = checkpoint[:POPULATION_SIZE,:,:]
    else: init_population = checkpoint

    population = init_population

    best_eval = 0; gen_detail = 0; fitness_hist = []
    elite_idx = np.array([], dtype=np.uint32)

    def detail_schedule(best_eval, gen_i, k=8.0, center=0.6):
        target = 1.0 / (1.0 + np.exp(-k * (best_eval - center)))
        target = 0.8 * target
        return target

    # region ------ MAIN LOOP ---------
    try:
        for gen_i in range(MAX_ITER):
            start_time = time.time()
            verbose = gen_i % 4 == 0

            gen_detail = max(detail_schedule(best_eval, gen_i), gen_detail)
            evals, ranks, (best_i,best_eval,best_img,best_img_heatmap) = evaluate_iter(targetimage,population,elite_idx,detail=gen_detail,iter_info=EVOLVE_ITER)

            eval_time = time.time() - start_time
            
            if verbose:
                print(f"Gen: {gen_i} | Best eval: {best_eval:.3f} | Detail: {gen_detail:.3f}")
                display_n_with_stats([best_img, targetimage, best_img_heatmap], wait=1, best_fitness=best_eval, fitness_hist=fitness_hist)

            display_time = time.time() - start_time - eval_time

            n_elite = int(max(1, POPULATION_SIZE * 1/32))
            population, elite_idx = select_and_mutate(population, ranks, elite_num=n_elite)
            
            select_and_mutate_time = time.time() - start_time - eval_time - display_time

            if verbose:
                print(f"Eval time:{eval_time:.3f} | S&M time:{select_and_mutate_time:.3f} | Display time:{display_time:.3f}")
    # endregion ------ END MAIN LOOP ---------

    except KeyboardInterrupt:
        print("KeyboardInterrupt. Exiting...")
    
    cv2.destroyAllWindows()

    try: save = input("Save? (y/n): ").lower() != "n"
    except EOFError: save = False
    if save: np.save(f"checkpoints/ch_{gen_i}.npy", population)

    return 0

