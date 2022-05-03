from enum import Enum

class DimReduceType(Enum):
    none = 0
    tsne = 1
    pca = 2
    tsvd = 3