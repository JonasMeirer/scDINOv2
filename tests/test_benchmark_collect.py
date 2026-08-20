"""Tests for aggregating benchmark results across the two pipeline stages.

Training and inference both write a file called `results.json`, with
overlapping metric keys, into different directories. The collector's job is to
recognise that those are two halves of one run rather than two runs.

Regression context: inference used to write into `outputs/inference/<date>/
<time>/`, disconnected from the sweep, so every purity/UMAP/HDBSCAN column in
the summary came back empty no matter what the pipeline computed.
"""

import json

import pandas as pd
import pytest

from scdino.benchmark.collect import aggregate, load_results

TRAIN_METRICS = {
    "val_knn_top1": "88.5000",
    "val_knn_top5": "99.1000",
    "val_silhouette": "0.0200",
}
INFER_METRICS = {
    "val_knn_top1": "90.0000",
    "val_knn_top5": "99.5000",
    "val_silhouette": "0.0300",
    "val_purity@1": "0.7000",
    "val_purity@10": "0.6800",
    "val_purity@100": "0.6600",
    "val_purity@1000": "0.6400",
    "val_purity_umap@1": "0.6900",
    "val_purity_umap@10": "0.6700",
    "val_purity_umap@100": "0.6500",
    "val_purity_umap@1000": "0.6300",
}


def write_run(root, arm, seed, *, train=True, inference=True, legacy=False):
    """Create one sweep-arm directory laid out as the pipeline writes it."""
    run = root / f"{arm}_seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    if train:
        doc = {"seed": seed, "metrics": dict(TRAIN_METRICS)}
        if not legacy:
            doc["stage"] = "train"
        else:
            doc |= {"config": {}, "runtime": {}}
        (run / "results.json").write_text(json.dumps(doc))
    if inference:
        (run / "inference").mkdir(exist_ok=True)
        doc = {"seed": seed, "metrics": dict(INFER_METRICS)}
        if not legacy:
            doc |= {"stage": "inference", "model_path": str(run / "hf_model")}
        (run / "inference" / "results.json").write_text(json.dumps(doc))
    return run


class TestMerging:
    def test_one_row_per_run_not_per_file(self, tmp_path):
        """Two results.json under one run must not become two rows."""
        write_run(tmp_path, "n1000", 42)
        rows = load_results(tmp_path)
        assert len(rows) == 1

    def test_inference_only_metrics_reach_the_summary(self, tmp_path):
        """The whole point: purity/UMAP columns must not come back empty."""
        write_run(tmp_path, "n1000", 42)
        (row,) = load_results(tmp_path)
        assert row["val_purity@1"] == "0.7000"
        assert row["val_purity_umap@1000"] == "0.6300"

    def test_inference_wins_on_overlapping_metrics(self, tmp_path):
        """Both stages report knn/silhouette; inference is the final word."""
        write_run(tmp_path, "n1000", 42)
        (row,) = load_results(tmp_path)
        assert row["val_knn_top1"] == INFER_METRICS["val_knn_top1"]
        assert row["val_silhouette"] == INFER_METRICS["val_silhouette"]

    def test_training_only_run_still_reported(self, tmp_path):
        write_run(tmp_path, "n1000", 42, inference=False)
        (row,) = load_results(tmp_path)
        assert row["has_train"] and not row["has_inference"]
        assert row["val_knn_top1"] == TRAIN_METRICS["val_knn_top1"]
        assert row["val_purity@1"] is None

    def test_inference_only_run_still_reported(self, tmp_path):
        write_run(tmp_path, "n1000", 42, train=False)
        (row,) = load_results(tmp_path)
        assert row["has_inference"] and not row["has_train"]
        assert row["val_purity@1"] == "0.7000"

    def test_legacy_files_without_a_stage_key_are_classified(self, tmp_path):
        """Results written before `stage` existed must still merge."""
        write_run(tmp_path, "n1000", 42, legacy=True)
        rows = load_results(tmp_path)
        assert len(rows) == 1
        assert rows[0]["has_train"] and rows[0]["has_inference"]
        assert rows[0]["val_purity@1"] == "0.7000"

    def test_multiple_arms_and_seeds(self, tmp_path):
        for arm in ("n100", "n1000"):
            for seed in (42, 43):
                write_run(tmp_path, arm, seed)
        rows = load_results(tmp_path)
        assert len(rows) == 4
        assert {r["experiment"] for r in rows} == {"n100", "n1000"}
        assert {r["seed"] for r in rows} == {42, 43}

    def test_empty_tree_yields_nothing(self, tmp_path):
        assert load_results(tmp_path) == []

    def test_malformed_json_is_skipped_not_fatal(self, tmp_path):
        write_run(tmp_path, "n1000", 42)
        bad = tmp_path / "broken_seed42"
        bad.mkdir()
        (bad / "results.json").write_text("{not json")
        rows = load_results(tmp_path)
        assert len(rows) == 1


class TestExperimentNaming:
    @pytest.mark.parametrize(
        "arm", ["n1000", "no_noise", "no_intensity_shift", "do_centercrop", "baseline"]
    )
    def test_arm_names_with_underscores_survive(self, tmp_path, arm):
        """Splitting on '_' and dropping the last token is not enough."""
        write_run(tmp_path, arm, 42)
        (row,) = load_results(tmp_path)
        assert row["experiment"] == arm

    def test_uses_the_recorded_seed_not_a_guess(self, tmp_path):
        write_run(tmp_path, "no_intensity_shift", 12345)
        (row,) = load_results(tmp_path)
        assert row["experiment"] == "no_intensity_shift"

    def test_directory_without_seed_suffix_keeps_its_name(self, tmp_path):
        run = tmp_path / "some_run"
        run.mkdir()
        (run / "results.json").write_text(
            json.dumps({"stage": "train", "seed": 42, "metrics": dict(TRAIN_METRICS)})
        )
        (row,) = load_results(tmp_path)
        assert row["experiment"] == "some_run"


class TestAggregation:
    def test_groups_seeds_into_one_row_per_arm(self, tmp_path):
        for arm in ("n100", "n1000"):
            for seed in (42, 43, 44):
                write_run(tmp_path, arm, seed)
        df = pd.DataFrame(load_results(tmp_path)).drop(
            columns=["run_dir", "has_train", "has_inference"]
        )
        df = df.astype(
            {c: float for c in df.columns if c != "experiment"}, errors="ignore"
        )
        agg = aggregate(df, "experiment")
        assert len(agg) == 2
        assert set(agg["experiment"]) == {"n100", "n1000"}
        assert (agg["n_seeds"] == 3).all()

    def test_purity_columns_are_aggregated_not_dropped(self, tmp_path):
        for seed in (42, 43):
            write_run(tmp_path, "n1000", seed)
        df = pd.DataFrame(load_results(tmp_path)).drop(
            columns=["run_dir", "has_train", "has_inference"]
        )
        df = df.astype(
            {c: float for c in df.columns if c != "experiment"}, errors="ignore"
        )
        agg = aggregate(df, "experiment")
        assert "val_purity@100_mean" in agg.columns
        assert agg["val_purity@100_mean"].iloc[0] == pytest.approx(0.66)


class TestModelDiscovery:
    """`benchmark.run` must hand stage 2 the run directory, not `hf_model/`.

    Regression: `find_hf_models` returned `p.parent` for
    `<run>/hf_model/config.json`, i.e. the `hf_model` directory itself. The
    caller then built `<run>/hf_model/hf_model`, so stage 2 could never load a
    checkpoint and every evaluation failed.
    """

    @staticmethod
    def make_run(root, name):
        run = root / name
        (run / "hf_model").mkdir(parents=True)
        (run / "hf_model" / "config.json").write_text("{}")
        (run / "results.json").write_text('{"stage": "train", "seed": 42}')
        return run

    def test_returns_run_dirs_not_checkpoint_dirs(self, tmp_path):
        from scdino.benchmark.run import find_hf_models

        self.make_run(tmp_path, "arm_seed42")
        (found,) = find_hf_models(tmp_path)
        assert found.name == "arm_seed42"
        assert found.name != "hf_model"

    def test_returned_path_can_be_used_to_build_the_checkpoint_path(self, tmp_path):
        """The caller does `model_dir / "hf_model"`; that must resolve."""
        from scdino.benchmark.run import find_hf_models

        self.make_run(tmp_path, "arm_seed42")
        (found,) = find_hf_models(tmp_path)
        assert (found / "hf_model" / "config.json").is_file()

    def test_finds_every_arm_and_sorts(self, tmp_path):
        from scdino.benchmark.run import find_hf_models

        for name in ("b_seed43", "a_seed42", "c_seed44"):
            self.make_run(tmp_path, name)
        assert [p.name for p in find_hf_models(tmp_path)] == [
            "a_seed42",
            "b_seed43",
            "c_seed44",
        ]

    def test_ignores_runs_without_an_exported_model(self, tmp_path):
        from scdino.benchmark.run import find_hf_models

        self.make_run(tmp_path, "good_seed42")
        (tmp_path / "failed_seed43").mkdir()
        (tmp_path / "failed_seed43" / "results.json").write_text("{}")
        assert [p.name for p in find_hf_models(tmp_path)] == ["good_seed42"]

    def test_evaluation_output_lands_inside_the_run(self, tmp_path):
        """Stage 2 writes `<run>/inference/`, which is what collect merges."""
        from scdino.benchmark.run import find_hf_models

        run = self.make_run(tmp_path, "arm_seed42")
        (found,) = find_hf_models(tmp_path)
        (found / "inference").mkdir()
        (found / "inference" / "results.json").write_text(
            json.dumps(
                {"stage": "inference", "seed": 42, "metrics": dict(INFER_METRICS)}
            )
        )
        (row,) = load_results(tmp_path)
        assert row["run_dir"] == str(run)
        assert row["has_train"] and row["has_inference"]
