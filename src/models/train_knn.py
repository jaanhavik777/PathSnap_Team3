"""K-nearest neighbors baseline for candidate scoring."""

from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_knn():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=15, weights="distance", algorithm="kd_tree", n_jobs=2),
    )
