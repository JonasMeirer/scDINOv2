import torch
import torch.nn.functional as F
from typing import Tuple


@torch.no_grad()
def knn_classifier(
    train_features,
    train_labels,
    test_features,
    k=20,
    T=0.07,
    num_classes=None,
    query_batch_size=1024,
    train_chunk_size=50000,
):
    """
    Memory-efficient k-NN classifier with two-stage top-k (like DINOv2, single GPU).

    Args:
        train_features: (N, D) tensor of reference features
        train_labels:   (N,) tensor of reference labels (int64)
        test_features:  (M, D) tensor of query features
        k: number of neighbors to vote with
        T: temperature for softmax weighting
        num_classes: number of classes (optional)
        query_batch_size: size of query minibatch
        train_chunk_size: size of train feature chunk
    Returns:
        probs: (M, C) tensor with class probabilities
        preds: (M,) tensor of predicted class indices
    """
    num_classes = num_classes or int(train_labels.max()) + 1

    # normalize once for cosine similarity
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    all_probs = []
    all_preds = []

    for q_start in range(0, len(test_features), query_batch_size):
        q_end = q_start + query_batch_size
        queries = test_features[q_start:q_end]  # (B, D)

        all_sims = []
        all_labels = []

        # loop over shards of the train set
        for t_start in range(0, len(train_features), train_chunk_size):
            t_end = t_start + train_chunk_size
            chunk_feats = train_features[t_start:t_end]  # (Nc, D)
            chunk_labels = train_labels[t_start:t_end]  # (Nc,)

            sims = queries @ chunk_feats.T  # (B, Nc)

            sims_topk, idx_topk = sims.topk(k, dim=1, largest=True, sorted=False)
            labels_topk = chunk_labels[idx_topk]  # (B, k)

            all_sims.append(sims_topk)
            all_labels.append(labels_topk)

        # concat local candidates
        all_sims = torch.cat(all_sims, dim=1)  # (B, k * n_chunks)
        all_labels = torch.cat(all_labels, dim=1)  # (B, k * n_chunks)

        # global top-k
        topk_sims, topk_idx = all_sims.topk(k, dim=1, largest=True, sorted=True)
        topk_labels = torch.gather(all_labels, 1, topk_idx)  # (B, k)

        weights = F.softmax(topk_sims / T, dim=1)  # (B, k)
        one_hot = F.one_hot(topk_labels, num_classes=num_classes).float()  # (B, k, C)
        probs = torch.sum(one_hot * weights.unsqueeze(-1), dim=1)  # (B, C)
        preds = probs.argmax(dim=1)  # (B,)

        all_probs.append(probs)
        all_preds.append(preds)

    all_probs = torch.cat(all_probs, dim=0)
    all_preds = torch.cat(all_preds, dim=0)
    return all_probs, all_preds


@torch.no_grad()
def compute_knn_accuracy(
    probs: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)
) -> dict:
    """
    Compute top-k accuracy for kNN predictions.

    Args:
        probs: (N, C) tensor with class probabilities
        targets: (N,) tensor with ground truth labels
        topk: tuple of top-k values to compute accuracy for

    Returns:
        dict: accuracy results for each k
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = probs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_k = correct_k.mul_(100.0 / batch_size).item()
        res[f"top{k}"] = acc_k

    return res


# ---- Example usage ----
if __name__ == "__main__":
    # fake dataset
    N, M, D, C = 5000, 200, 128, 10  # train, test, dim, classes
    train_x = torch.randn(N, D, device="cuda")
    train_y = torch.randint(0, C, (N,), device="cuda")
    test_x = torch.randn(M, D, device="cuda")
    test_y = torch.randint(0, C, (M,), device="cuda")

    probs = knn_classifier(
        train_x,
        train_y,
        test_x,
        k=20,
        T=0.07,
        num_classes=C,
        query_batch_size=1024,
        train_chunk_size=50000,
    )
    results = compute_knn_accuracy(probs, test_y, topk=(1, 5))

    print(f"kNN top1 accuracy: {results['top1']:.2f}%")
    print(f"kNN top5 accuracy: {results['top5']:.2f}%")
