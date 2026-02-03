import numpy as np

from src.genotype import Genotype

# region Parent Selection
def select_parents(rankings: np.ndarray) -> np.ndarray:
    N = len(rankings)
    mask = tournament_selection_mask(rankings, target_n=int(N / 3))
    return mask

def tournament_selection_mask(rankings: np.ndarray, target_n: int, k: int=2) -> np.ndarray:
    N = len(rankings)
    selected = np.zeros(N, dtype=bool)

    for _ in range(target_n):
        idx = np.random.choice(N, size=k, replace=False)
        winner = idx[np.argmin(rankings[idx])]  # lower rank = better
        selected[winner] = True

    return selected
# endregion

# region Crossover
def crossover(parents, num_children):
    return uniform_crossover(parents,num_children)

def uniform_crossover(parents, num_children):
    P, G, D = parents.shape

    if P < 2: raise ValueError("crossover: need at least 2 parents")
    if num_children < 1: raise ValueError("crossover: need num_children > 0")

    # pick two parents per child
    p1 = np.random.randint(0, P, size=num_children)
    p2 = np.random.randint(0, P, size=num_children)

    a = parents[p1]  # (M, G, D)
    b = parents[p2]  # (M, G, D)

    # per-gene mask (whole 7-vector copied from one parent or the other)
    m = (np.random.rand(num_children, G, 1) < 0.5)
    children = np.where(m, a, b)

    return children
# endregion


# region Mutation
def mutate(population: np.ndarray) -> np.ndarray:
    N, G, D = population.shape # population, num_genes, things_per_gene

    pop_rate=0.3; 

    jitter_rate=0.05; sigma=0.05; sigma_idx=0.01
    respawn_rate=0.005; kill_rate=0.005

    def param_jitter(gene: np.ndarray):
        if np.random.rand() > jitter_rate:
            return gene
        
        noise = np.random.normal(0.0, sigma, size=(D-1))
        idx_noise = np.random.normal(0.0, sigma_idx, size=(1))
        return gene + np.concatenate([noise,idx_noise])
    
    def respawn(gene: np.ndarray):
        return np.random.rand(D) if np.random.rand() < respawn_rate else gene

    def kill(gene: np.ndarray):
        return np.zeros(D) if np.random.rand() < kill_rate else gene

    x = population.copy()

    # which individuals mutate
    mutate_mask = np.random.rand(N) < pop_rate

    for i in np.where(mutate_mask)[0]:
        # number of genes to mutate in this child
        n_mut = np.random.randint(1, 6)  # 1–5 genes
        gene_idx = np.random.choice(G, size=n_mut, replace=False)

        for g in gene_idx:
            gene = x[i, g]
            gene = param_jitter(gene)
            gene = respawn(gene)
            gene = kill(gene)
            x[i, g] = gene

        # keep gene-order invariance
        #perm = np.random.permutation(G)
        #x[i] = x[i, perm]

    np.clip(x, 0.0, 1.0, out=x)
    return x
# endregion