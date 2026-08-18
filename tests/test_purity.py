"""Tests for the neighbourhood-purity metrics used in inference reporting."""

import numpy as np
import pytest

from src.scdino.eval.purity import (
    compute_neighbor_labels,
    compute_purity,
    purity_at_k,
    purity_per_class,
)


@pytest.fixture
def separated_two_class_data():
    """Two tight, far-apart clusters: every near neighbour shares a label."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.01, size=(30, 2))
    b = rng.normal(50.0, 0.01, size=(30, 2))
    features = np.vstack([a, b])
    labels = np.array([0] * 30 + [1] * 30)
    return features, labels


class TestComputeNeighborLabels:
    def test_excludes_the_point_itself(self, separated_two_class_data):
        features, labels = separated_two_class_data
        same = compute_neighbor_labels(features, labels, max_k=5)
        assert same.shape == (60, 5)
        assert same.dtype == bool

    def test_perfectly_separated_clusters_are_pure(self, separated_two_class_data):
        features, labels = separated_two_class_data
        same = compute_neighbor_labels(features, labels, max_k=10)
        assert same.all()

    def test_interleaved_labels_are_impure(self):
        """Points on a line with alternating labels have impure neighbourhoods."""
        features = np.arange(40, dtype=float).reshape(-1, 1)
        labels = np.arange(40) % 2
        same = compute_neighbor_labels(features, labels, max_k=2)
        # The two nearest neighbours of an interior point are its immediate
        # line neighbours, which always carry the opposite label.
        assert not same[1:-1].any()


class TestPurityAtK:
    def test_is_the_fraction_of_matching_neighbours(self):
        same = np.array([[True, True, False, False]])
        assert purity_at_k(same, 4)[0] == pytest.approx(0.5)
        assert purity_at_k(same, 2)[0] == pytest.approx(1.0)

    def test_returns_one_value_per_sample(self):
        same = np.ones((7, 5), dtype=bool)
        assert purity_at_k(same, 5).shape == (7,)


class TestComputePurity:
    def test_perfect_separation_gives_purity_one(self, separated_two_class_data):
        features, labels = separated_two_class_data
        _, purities = compute_purity(features, labels, ks=(1, 5, 10))
        assert set(purities) == {1, 5, 10}
        for value in purities.values():
            assert value == pytest.approx(1.0)

    def test_purity_lies_in_the_unit_interval(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(80, 4))
        labels = rng.integers(0, 3, size=80)
        _, purities = compute_purity(features, labels, ks=(1, 5))
        for value in purities.values():
            assert 0.0 <= value <= 1.0

    def test_random_labels_approach_the_chance_rate(self):
        rng = np.random.default_rng(0)
        features = rng.normal(size=(400, 4))
        labels = rng.integers(0, 4, size=400)  # chance purity = 1/4
        _, purities = compute_purity(features, labels, ks=(20,))
        assert purities[20] == pytest.approx(0.25, abs=0.1)


class TestPurityPerClass:
    def test_reports_one_entry_per_class(self, separated_two_class_data):
        features, labels = separated_two_class_data
        same, _ = compute_purity(features, labels, ks=(5,))
        per_class = purity_per_class(same, labels, k=5)
        assert set(per_class) == {0, 1}
        for value in per_class.values():
            assert value == pytest.approx(1.0)

    def test_accepts_string_class_names(self, separated_two_class_data):
        """inference/run.py passes human-readable class names, not integers."""
        features, labels = separated_two_class_data
        same, _ = compute_purity(features, labels, ks=(5,))
        names = np.where(labels == 0, "alpha", "beta")
        per_class = purity_per_class(same, names, k=5)
        assert set(per_class) == {"alpha", "beta"}
