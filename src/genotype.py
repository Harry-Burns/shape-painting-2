
import numpy as np

class Genotype:
    # Species consists of a list of "GENES"
    #   1. There is a maximum number of genes
    #   2. Each gene consists of 3 pieces of information, each of which is stored from 0-1
    #       2.1. Size, w/h ratio, position (x, y), color (greyscale/rgb + opacity), index drawing order
    #       2.2. This means each gene (row) in the np array will be 1+1+2+2+1 long

    max_genes = 64
    gene_shape = (max_genes,7)

    def __init__(self, genes: np.ndarray=None):
        if genes is None: genes = np.random.rand(Genotype.gene_shape[0], Genotype.gene_shape[1])
        assert genes.shape == Genotype.gene_shape
        self.genes = genes

    