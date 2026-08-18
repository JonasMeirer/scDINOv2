"""Every shipped Hydra config must compose.

These are cheap guards against config rot: a renamed key, a `defaults` entry
pointing at a file that no longer exists, or an experiment override that no
longer matches the model it overrides. None of this needs a GPU or a dataset.
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = (Path(__file__).resolve().parents[1] / "configs").as_posix()
CONFIG_ROOT = Path(CONFIG_DIR)


def _options(group: str) -> list[str]:
    """Hydra option names for a config group, e.g. 'pretrained/dinov2-large'."""
    group_dir = CONFIG_ROOT / group
    if not group_dir.is_dir():
        return []
    return sorted(
        p.relative_to(group_dir).with_suffix("").as_posix()
        for p in group_dir.rglob("*.yaml")
    )


MODELS = _options("model")
EXPERIMENTS = _options("experiment")
DATAMODULES = _options("datamodule")
LOGGERS = _options("logging")


@pytest.fixture(autouse=True)
def _data_dir(monkeypatch, tmp_path):
    """The datamodule configs interpolate ${oc.env:DATA_DIR}."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


def compose_config(config_name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(config_name=config_name, overrides=overrides)


class TestConfigGroupsAreDiscoverable:
    def test_expected_groups_are_not_empty(self):
        assert MODELS, "no model configs found"
        assert EXPERIMENTS, "no experiment configs found"
        assert DATAMODULES, "no datamodule configs found"


class TestRootConfigs:
    def test_train_config_composes(self):
        cfg = compose_config("train", [])
        assert cfg.mode == "train"
        assert cfg.model is not None
        assert cfg.datamodule is not None
        assert cfg.trainer is not None

    def test_inference_config_composes(self):
        cfg = compose_config("inference", [])
        assert cfg.mode == "inference"
        assert cfg.eval is not None

    def test_inference_ships_no_machine_specific_checkpoint_path(self):
        """local_model_path must not point at a developer's filesystem."""
        cfg = compose_config("inference", [])
        assert cfg.local_model_path in (None, ""), (
            "configs/inference.yaml must default local_model_path to null"
        )


class TestModelConfigs:
    @pytest.mark.parametrize("model", MODELS)
    def test_composes_with_train(self, model):
        cfg = compose_config("train", [f"model={model}"])
        assert cfg.model.name

    @pytest.mark.parametrize("model", MODELS)
    def test_has_a_matching_transform_config(self, model):
        """datamodule/chronotype.yaml resolves transforms from the model name."""
        cfg = compose_config("train", [f"model={model}"])
        assert cfg.datamodule.transforms is not None

    @pytest.mark.parametrize("model", [m for m in MODELS if not m.startswith("pretrained/")])
    def test_trainable_models_declare_a_lightning_target(self, model):
        cfg = compose_config("train", [f"model={model}"])
        assert "_target_" in cfg.model, f"{model} has no _target_ to instantiate"
        assert "training" in cfg.model, f"{model} has no training block"


class TestExperimentConfigs:
    @pytest.mark.parametrize("experiment", EXPERIMENTS)
    def test_composes_with_train(self, experiment):
        cfg = compose_config("train", [f"+experiment={experiment}"])
        assert cfg.model is not None

    @pytest.mark.parametrize("experiment", EXPERIMENTS)
    def test_declares_a_sweep_output_directory(self, experiment):
        """Experiments are run with --multirun, so hydra.sweep.dir matters."""
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="train",
                overrides=[f"+experiment={experiment}"],
                return_hydra_config=True,
            )
            assert cfg.hydra.sweep.dir

    @pytest.mark.parametrize(
        "experiment", [e for e in EXPERIMENTS if e.startswith("augmentation/")]
    )
    def test_augmentation_ablations_actually_change_a_transform(self, experiment):
        """An ablation that resolves to the baseline config measures nothing."""
        baseline = compose_config("train", [])
        ablated = compose_config("train", [f"+experiment={experiment}"])
        assert ablated.datamodule.transforms != baseline.datamodule.transforms, (
            f"experiment/{experiment} leaves datamodule.transforms unchanged"
        )


class TestDatamoduleConfigs:
    @pytest.mark.parametrize("datamodule", DATAMODULES)
    def test_composes(self, datamodule):
        if datamodule.startswith("transforms/"):
            pytest.skip("transform fragments are composed via the datamodule, not directly")
        cfg = compose_config("train", [f"datamodule={datamodule}"])
        assert cfg.datamodule.paths.train_dir
        assert cfg.datamodule.paths.val_dir

    def test_normalisation_statistics_match_the_channel_count(self):
        cfg = compose_config("train", [])
        loader = cfg.datamodule.loader
        n = loader.num_channels
        stats = loader.norm_dict[loader.norm_type]
        assert len(stats.mean) == n
        assert len(stats.std) == n
        if loader.max_vals_clip is not None:
            assert len(loader.max_vals_clip) == n

    def test_every_normalisation_preset_is_fully_specified(self):
        cfg = compose_config("train", [])
        loader = cfg.datamodule.loader
        for name, stats in loader.norm_dict.items():
            assert len(stats.mean) == loader.num_channels, f"{name}.mean"
            assert len(stats.std) == loader.num_channels, f"{name}.std"
            assert all(s > 0 for s in stats.std), f"{name}.std has a non-positive entry"


class TestLoggingConfigs:
    @pytest.mark.parametrize("logger", LOGGERS)
    def test_composes_and_names_itself(self, logger):
        cfg = compose_config("train", [f"logging={logger}"])
        assert cfg.logging.name in {"console", "wandb", "mlflow"}


class TestReferentialIntegrity:
    def test_no_config_references_a_removed_strucperc_variant(self):
        """StrucPerc was removed; stale references would fail at instantiation."""
        offenders = [
            p.relative_to(CONFIG_ROOT).as_posix()
            for p in CONFIG_ROOT.rglob("*.yaml")
            if "StrucPerc" in p.read_text()
        ]
        assert not offenders, f"stale StrucPerc references in {offenders}"

    def test_every_config_is_valid_yaml(self):
        for path in CONFIG_ROOT.rglob("*.yaml"):
            OmegaConf.load(path)
