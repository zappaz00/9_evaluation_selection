from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from .defs import ModelType, DimReduceType
from typing import Any


def create_pipeline(
    use_dim_red: DimReduceType,
    use_scaler: bool,
    model_type: ModelType,
    hyperparams: dict[str, Any],
    random_state: int,
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    reductor = None
    n_components = hyperparams.pop("n_components", None)
    if use_dim_red == DimReduceType.pca:
        if n_components is not None:
            reductor = PCA(n_components=n_components)
        else:
            reductor = PCA()
    elif use_dim_red == DimReduceType.tsvd:
        if n_components is not None:
            reductor = TruncatedSVD(n_components=n_components)
        else:
            reductor = TruncatedSVD()

    if reductor is not None:
        pipeline_steps.append(("reductor", reductor))

    clf = None
    if model_type == ModelType.logreg:
        clf = LogisticRegression(random_state=random_state)
    elif model_type == ModelType.randomforest:
        clf = RandomForestClassifier(random_state=random_state)
    elif model_type == ModelType.knn:
        clf = KNeighborsClassifier()  # no random_state
    elif model_type is None:
        clf = None
    else:
        raise ValueError("Unknown model type")

    if clf is not None:
        clf.set_params(**hyperparams)
        pipeline_steps.append(("classifier", clf))

    return Pipeline(steps=pipeline_steps)
