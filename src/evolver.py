import time
import numpy as np
import cv2

from src.genotype import Genotype
from src.phenotype import phenotype_population

from src.evaluate import evaluate_iter, evaluate
from src.selection import select_parents, crossover, mutate
from src.display import display_n_with_stats

POPULATION_SIZE = 256
MAX_ITER = 64000

#EVOLVE_ITER = {
#    'quality': [0.25, 1],
#    'keep': [0.25, 1],
#    'method': ["ssim", "ssim"]
#}

def init_population(checkpoint: np.ndarray) -> np.ndarray:
    init_population = np.random.rand(POPULATION_SIZE, Genotype.gene_shape[0], Genotype.gene_shape[1]);

    if checkpoint is None: 
        pass; #init_population[:,:,5:6] = 0 # optional init params
    elif checkpoint.shape[1:] != Genotype.gene_shape: 
        raise ValueError(f"evolve: checkpoint size was wrong. Expected {init_population.shape} got {checkpoint.shape}")
    elif checkpoint.shape[0] != POPULATION_SIZE:
        if checkpoint.shape[0] < POPULATION_SIZE:
            print("Checkpoint population size too small, adding random population...")
            init_population[:checkpoint.shape[0],:,:] = checkpoint[:,:,:]
        else:
            print("Checkpoint population size too large, chopping off the end...")
            init_population[:,:,:] = checkpoint[:POPULATION_SIZE,:,:]
    else: 
        init_population = checkpoint

    return init_population


def evolve(targetimage: np.ndarray, checkpoint: np.ndarray | None = None) -> np.ndarray:
    #   assert len(EVOLVE_ITER['quality']) == len(EVOLVE_ITER['keep']) == len(EVOLVE_ITER['method'])
    h,w = targetimage.shape[:2]

    population = init_population(checkpoint)
    phenotypes = np.zeros([POPULATION_SIZE,h,w], dtype=np.uint8)
    evaluations = np.zeros(POPULATION_SIZE, dtype=float)
    calc_mask = np.ones(POPULATION_SIZE, dtype=bool)

    pht=time.time()
    phenotype_population(population,phenotypes,image_data=(w,h),mask=calc_mask)
    print(f"Initial phenotype generation time: {(time.time()-pht):.4f}")

    evst=time.time()
    rankings, best = evaluate(targetimage,phenotypes,evaluations,calc_mask, method = 'ssim')
    best_i,best_eval,best_img,best_img_heatmap = best
    print(f"Initial eval time: {(time.time()-evst):.4f}")

    fitness_hist = []

    def detail_schedule():
        return 1

    # region ------ MAIN LOOP ---------
    try:
        for gen_i in range(MAX_ITER):
            verbose = gen_i % 4 == 0
            gen_detail = detail_schedule()

            keep = max(1, int(POPULATION_SIZE * 1/32)); remove = POPULATION_SIZE - keep

            sst = time.time()
            parents_mask = select_parents(rankings)
            children = crossover(population[parents_mask], remove)
            children = mutate(children)
            sst = time.time() - sst

            # this is technically using elites still --- Cut off bottom X values from population and replace with children
            

            calc_mask[:] = False
            order = np.argsort(evaluations)[::-1]
            remove_idx = order[keep:]

            population[remove_idx] = children
            calc_mask[remove_idx] = True

            pht = time.time()
            phenotype_population(population, phenotypes, image_data=(w, h), mask=calc_mask, detail=gen_detail)
            pht = time.time() - pht

            evst = time.time()
            rankings, best = evaluate(targetimage,phenotypes,evaluations,calc_mask,method='ssim')
            best_i, best_eval, best_img, best_img_heatmap = best
            evst = time.time() - evst

            fitness_hist.append(best_eval)


            dst = time.time()
            if verbose:
                print(f"Gen: {gen_i} | Best eval: {best_eval:.3f} | Detail: {gen_detail:.3f}")
                display_n_with_stats([best_img, targetimage, best_img_heatmap], wait=1, best_fitness=best_eval, fitness_hist=fitness_hist)
            dst = time.time() - dst

            if verbose:
                print(f"Phenotype time:{pht:.3f} | Eval time:{evst:.3f} | S&M time:{sst:.3f} | Display time:{dst:.3f}")
    # endregion ------ END MAIN LOOP ---------

    except KeyboardInterrupt:
        print("KeyboardInterrupt. Exiting...")
    
    cv2.destroyAllWindows()

    try: save = input("Save? (y/n): ").lower() != "n"
    except EOFError: save = False
    if save: np.save(f"checkpoints/ch_{gen_i}.npy", population)

    return 0

