from enum import Enum


class DimReduceType(Enum):
    none = 0
    pca = 2
    tsvd = 3


class ModelType(Enum):
    logreg = 1
    knn = 2
    randomforest = 3


class TuneType(Enum):
    auto_random = 0
    auto_grid = 1
    manual = 2
