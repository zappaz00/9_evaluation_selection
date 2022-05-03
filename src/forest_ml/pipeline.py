from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from .model_type import ModelType
from .dim_red_type import DimReduceType

def create_pipeline(
    use_dim_red: DimReduceType, dim_red_comp: int, use_scaler: bool, model_type: ModelType, hyperparams: dict, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

    reductor = None
    if use_dim_red == DimReduceType.pca:
        reductor = PCA(n_components=dim_red_comp)
    elif use_dim_red == DimReduceType.tsvd:
        reductor = TruncatedSVD(n_components=dim_red_comp)
    elif use_dim_red == DimReduceType.tsne:
        reductor = TSNE(n_components=dim_red_comp)

    if reductor is not None:
        pipeline_steps.append(
            (
                "reductor", reductor
            )
        )

    clf = None
    if model_type == ModelType.logreg:
        clf = LogisticRegression(random_state=random_state)
    elif model_type == ModelType.randomforest:
        clf = RandomForestClassifier(random_state=random_state)
    elif model_type == ModelType.knn:
        clf = KNeighborsClassifier() #no random_state
    else:
        raise ValueError('Unknown model type')

    clf.set_params(**hyperparams)
    pipeline_steps.append(
        (
            "classifier", clf
        )
    )

    return Pipeline(steps=pipeline_steps)