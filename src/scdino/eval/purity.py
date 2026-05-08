import numpy as np
from sklearn.neighbors import NearestNeighbors


def compute_neighbor_labels(features, labels, max_k):
    """Fit NN on features and return a (N, max_k) bool array of neighbor-label matches (self excluded)."""
    nn = NearestNeighbors(n_neighbors=max_k + 1, algorithm="auto")
    nn.fit(features)
    _, indices = nn.kneighbors(features)
    indices = indices[:, 1:]
    return labels[indices] == labels[:, None]


def purity_at_k(same_label, k):
    return same_label[:, :k].mean(axis=1)


def compute_purity(features, labels, ks=(1, 10, 100, 1000)):
    """Compute mean purity@k for each k in ks.

    Returns:
        same_label: (N, max(ks)) bool array of neighbor-label matches
        purities: dict mapping k -> mean purity@k
    """
    same_label = compute_neighbor_labels(features, labels, max_k=max(ks))
    purities = {k: float(purity_at_k(same_label, k).mean()) for k in ks}
    return same_label, purities


def purity_per_class(same_label, labels, k):
    """Return dict mapping class label -> mean purity@k restricted to samples of that class."""
    out = {}
    for cls in np.unique(labels):
        mask = labels == cls
        out[cls] = float(purity_at_k(same_label, k)[mask].mean())
    return out
