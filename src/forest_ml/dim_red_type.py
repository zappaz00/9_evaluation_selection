from enum import Enum

class DimReduceType(Enum):
    NONE = 0
    TSNE = 1
    PCA = 2
    TSVD = 3