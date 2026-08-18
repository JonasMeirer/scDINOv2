"""Tests for the kNN classifier used by every evaluation path.

The chunked two-stage search in `knn_classifier` is the piece most likely to
drift, so it is checked against a brute-force reference rather than against
hardcoded numbers.
"""

import pytest
import torch
import torch.nn.functional as F

from src.scdino.eval.knn import compute_knn_accuracy, knn_classifier


def brute_force_knn(train_features, train_labels, test_features, k, T, num_classes):
    """Unchunked reference implementation of the same algorithm."""
    train = F.normalize(train_features, dim=1)
    test = F.normalize(test_features, dim=1)
    sims = test @ train.T
    topk_sims, topk_idx = sims.topk(k, dim=1, largest=True, sorted=True)
    topk_labels = train_labels[topk_idx]
    weights = F.softmax(topk_sims / T, dim=1)
    one_hot = F.one_hot(topk_labels, num_classes=num_classes).float()
    return (one_hot * weights.unsqueeze(-1)).sum(dim=1)


class TestKnnClassifier:
    def test_matches_brute_force_reference(self):
        torch.manual_seed(0)
        train_x = torch.randn(300, 16)
        train_y = torch.randint(0, 4, (300,))
        test_x = torch.randn(50, 16)

        probs, _ = knn_classifier(
            train_x, train_y, test_x, k=10, T=0.07, num_classes=4,
            query_batch_size=16, train_chunk_size=64,
        )
        expected = brute_force_knn(train_x, train_y, test_x, k=10, T=0.07, num_classes=4)
        torch.testing.assert_close(probs, expected, rtol=1e-4, atol=1e-5)

    @pytest.mark.parametrize("train_chunk_size", [64, 128, 512])
    @pytest.mark.parametrize("query_batch_size", [8, 32, 256])
    def test_chunking_does_not_change_the_answer(self, train_chunk_size, query_batch_size):
        """Chunk sizes are a memory knob and must never affect the result."""
        torch.manual_seed(0)
        train_x, train_y = torch.randn(256, 8), torch.randint(0, 3, (256,))
        test_x = torch.randn(40, 8)

        reference, _ = knn_classifier(
            train_x, train_y, test_x, k=5, T=0.07, num_classes=3,
            query_batch_size=256, train_chunk_size=256,
        )
        probs, _ = knn_classifier(
            train_x, train_y, test_x, k=5, T=0.07, num_classes=3,
            query_batch_size=query_batch_size, train_chunk_size=train_chunk_size,
        )
        torch.testing.assert_close(probs, reference, rtol=1e-4, atol=1e-5)

    def test_separable_classes_are_classified_perfectly(self, labelled_features):
        feats, labels = labelled_features
        probs, preds = knn_classifier(feats, labels, feats, k=5, T=0.07, num_classes=3)
        assert (preds == labels).all()

    def test_probabilities_form_a_distribution(self):
        torch.manual_seed(0)
        train_x, train_y = torch.randn(100, 8), torch.randint(0, 5, (100,))
        probs, preds = knn_classifier(
            train_x, train_y, torch.randn(20, 8), k=7, T=0.07, num_classes=5,
        )
        assert probs.shape == (20, 5)
        assert (probs >= 0).all()
        torch.testing.assert_close(probs.sum(dim=1), torch.ones(20), rtol=1e-5, atol=1e-6)
        assert (preds == probs.argmax(dim=1)).all()

    def test_handles_a_final_train_chunk_smaller_than_k(self):
        """The last chunk can hold fewer than k rows; k is clamped per chunk.

        Regression test for a crash whenever
        `len(train_features) % train_chunk_size < k`.
        """
        train_x, train_y = torch.randn(105, 8), torch.randint(0, 3, (105,))
        probs, _ = knn_classifier(
            train_x, train_y, torch.randn(4, 8), k=20, T=0.07,
            num_classes=3, train_chunk_size=100,
        )
        assert probs.shape == (4, 3)

    def test_ragged_chunking_still_matches_the_reference(self):
        """Clamping k per chunk must not change the answer."""
        torch.manual_seed(0)
        train_x, train_y = torch.randn(105, 8), torch.randint(0, 3, (105,))
        test_x = torch.randn(6, 8)
        expected = brute_force_knn(train_x, train_y, test_x, k=20, T=0.07, num_classes=3)
        probs, _ = knn_classifier(
            train_x, train_y, test_x, k=20, T=0.07,
            num_classes=3, train_chunk_size=100,
        )
        torch.testing.assert_close(probs, expected, rtol=1e-4, atol=1e-5)

    def test_k_larger_than_the_reference_set_is_clamped(self):
        train_x, train_y = torch.randn(5, 4), torch.randint(0, 2, (5,))
        probs, _ = knn_classifier(train_x, train_y, torch.randn(3, 4), k=50, T=0.07, num_classes=2)
        assert probs.shape == (3, 2)

    def test_rejects_an_empty_reference_set(self):
        with pytest.raises(ValueError, match="at least one reference sample"):
            knn_classifier(
                torch.zeros(0, 4), torch.zeros(0, dtype=torch.long),
                torch.randn(2, 4), k=5, T=0.07, num_classes=2,
            )

    def test_infers_num_classes_from_labels_when_not_given(self):
        train_x = torch.randn(50, 4)
        train_y = torch.randint(0, 6, (50,))
        train_y[0] = 5  # guarantee the top class is present
        probs, _ = knn_classifier(train_x, train_y, torch.randn(5, 4), k=3, T=0.07)
        assert probs.shape[1] == 6

    def test_lower_temperature_sharpens_the_vote(self):
        torch.manual_seed(0)
        train_x, train_y = torch.randn(200, 8), torch.randint(0, 4, (200,))
        test_x = torch.randn(20, 8)
        sharp, _ = knn_classifier(train_x, train_y, test_x, k=10, T=0.01, num_classes=4)
        soft, _ = knn_classifier(train_x, train_y, test_x, k=10, T=1.0, num_classes=4)
        assert sharp.max(dim=1).values.mean() > soft.max(dim=1).values.mean()


class TestComputeKnnAccuracy:
    def test_perfect_and_zero_accuracy(self):
        probs = torch.eye(4)
        assert compute_knn_accuracy(probs, torch.arange(4), topk=(1,))["top1"] == 100.0
        rolled = torch.roll(torch.eye(4), shifts=1, dims=1)
        assert compute_knn_accuracy(rolled, torch.arange(4), topk=(1,))["top1"] == 0.0

    def test_known_partial_accuracy(self):
        # 3 of 4 predictions correct.
        probs = torch.eye(4)
        targets = torch.tensor([0, 1, 2, 0])
        assert compute_knn_accuracy(probs, targets, topk=(1,))["top1"] == pytest.approx(75.0)

    def test_topk_is_monotone_in_k(self):
        torch.manual_seed(0)
        probs = torch.rand(200, 10)
        res = compute_knn_accuracy(probs, torch.randint(0, 10, (200,)), topk=(1, 5))
        assert res["top5"] >= res["top1"]

    def test_drops_k_larger_than_the_number_of_classes(self):
        """top-5 is undefined for a 3-class problem and must not be reported."""
        res = compute_knn_accuracy(torch.rand(10, 3), torch.randint(0, 3, (10,)), topk=(1, 5))
        assert "top5" not in res
        assert "top1" in res

    def test_drops_k_equal_to_the_number_of_classes(self):
        """top-5 on a 5-class problem is trivially 100% and must be dropped."""
        res = compute_knn_accuracy(torch.rand(50, 5), torch.randint(0, 5, (50,)), topk=(1, 5))
        assert "top5" not in res
        assert "top1" in res

    def test_keeps_k_strictly_below_the_number_of_classes(self):
        res = compute_knn_accuracy(torch.rand(50, 6), torch.randint(0, 6, (50,)), topk=(1, 5))
        assert "top5" in res

    def test_always_reports_something(self):
        """Even if every requested k is dropped, top1 is reported."""
        res = compute_knn_accuracy(torch.rand(10, 2), torch.randint(0, 2, (10,)), topk=(5,))
        assert set(res) == {"top1"}
