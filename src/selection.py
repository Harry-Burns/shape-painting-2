import numpy as np

from src.genotype import Genotype

def select_and_mutate(population: np.ndarray, rankings: np.ndarray, elite_num: int=0) -> np.ndarray:
    if population.shape[0] != len(rankings):    raise ValueError(f"select_and_mutate: population and evaluation lists must be the same length. Received {population.shape[0]}, {len(rankings)}")
    if elite_num > population.shape[0]:         raise ValueError(f"Can't have more elites than population size. Received pop: {population.shape[0]}, elite_num: {elite_num}")
    
    N = population.shape[0]


    elite_idx = np.argpartition(rankings, elite_num)[:elite_num]
    elites = population[elite_idx] if elite_num > 0 else population[:0]
    elite_idx = np.arange(elites.shape[0], dtype=np.uint32) if elite_num > 0 else None    

    
    parents = population[tournament_selection_mask(rankings, n_select= int(N / 3))]

    if parents.shape[0] < 2: parents = population[np.argpartition(rankings, 1)[:2]]; print("Not enough parents selected, instead using the top 2 as parents.")
    
    children = crossover(parents, population_size=(N - elite_num))
    children = mutation(children)

    population_out = children if elite_num == 0 else np.concatenate([elites, children], axis=0)

    return population_out, elite_idx


def tournament_selection_mask(rankings: np.ndarray, n_select: int, k: int=2) -> np.ndarray:
    N = len(rankings)
    selected = np.zeros(N, dtype=bool)

    for _ in range(n_select):
        idx = np.random.choice(N, size=k, replace=False)
        winner = idx[np.argmin(rankings[idx])]  # lower rank = better
        selected[winner] = True

    return selected

def crossover(parents, population_size):
    P, G, D = parents.shape

    if P < 2: raise ValueError("crossover: need at least 2 parents")

    # pick two parents per child
    p1 = np.random.randint(0, P, size=population_size)
    p2 = np.random.randint(0, P, size=population_size)

    a = parents[p1]  # (M, G, D)
    b = parents[p2]  # (M, G, D)

    # per-gene mask (whole 7-vector copied from one parent or the other)
    m = (np.random.rand(population_size, G, 1) < 0.5)
    children = np.where(m, a, b)

    # since gene order is irrelevant, randomise gene order in each child
    # (prevents positional bias from being learned accidentally)
    out = np.empty_like(children)
    for i in range(population_size):
        out[i] = children[i]

    return out

def mutation(population: np.ndarray) -> np.ndarray:
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
        perm = np.random.permutation(G)
        x[i] = x[i, perm]

    np.clip(x, 0.0, 1.0, out=x)
    return x