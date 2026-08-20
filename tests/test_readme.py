"""The README must stay executable.

Documentation rots silently: a renamed config or a changed flag breaks the
commands people copy out of the README, and nothing fails until a newcomer
tries them. These tests parse the README's own bash blocks and check every
claim they make against the real tree.

They are deliberately static — no training is launched. What they catch is the
class of defect the README actually had: entry points that no longer exist,
Hydra keys that were never real, config groups referenced by names that do not
exist on disk, and directories documented but never created.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
CONFIGS = REPO / "configs"


@pytest.fixture(scope="module")
def commands() -> list[str]:
    """Every runnable line inside a ```bash block, with continuations joined."""
    text = README.read_text()
    out = []
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        joined = re.sub(r"\\\n\s*", " ", block)
        for line in joined.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(line)
    return out


def test_readme_exists_and_has_commands(commands):
    assert README.is_file()
    assert len(commands) >= 8, "expected the README to document several commands"


class TestEntryPoints:
    def test_every_documented_module_is_importable(self, commands):
        """`python -m scdino.x.y` must name a module that actually exists."""
        import importlib.util

        modules = sorted(
            {m for c in commands for m in re.findall(r"python -m (scdino[\w.]*)", c)}
        )
        assert modules, "no `python -m scdino.*` commands found"
        for module in modules:
            assert importlib.util.find_spec(module) is not None, (
                f"README documents `python -m {module}`, which does not exist"
            )

    def test_no_stale_src_prefixed_invocations(self):
        """The package is imported as `scdino`, never `src.scdino`."""
        assert "src.scdino" not in README.read_text(), (
            "README references the old `src.scdino` module path"
        )

    def test_no_script_path_invocations_of_package_modules(self, commands):
        """`python src/scdino/...py` bypasses the installed package."""
        offenders = [c for c in commands if re.search(r"python\s+src/scdino/", c)]
        assert not offenders, f"use `python -m scdino...` instead: {offenders}"


class TestHydraOverrides:
    """Every `group=option` override must name a config that exists."""

    GROUPS = ("model", "datamodule", "logging", "trainer")

    def _overrides(self, commands, group):
        found = set()
        for cmd in commands:
            if not re.search(r"python -m scdino\.(train|inference)", cmd):
                continue
            for match in re.findall(rf"(?<![\w.]){group}=([\w/.-]+)", cmd):
                found.add(match)
        return found

    @pytest.mark.parametrize("group", GROUPS)
    def test_config_group_options_exist(self, commands, group):
        for option in self._overrides(commands, group):
            if option in ("null", "None"):
                continue
            assert (CONFIGS / group / f"{option}.yaml").is_file(), (
                f"README uses `{group}={option}` but "
                f"configs/{group}/{option}.yaml does not exist"
            )

    def test_experiment_references_exist(self):
        """Every name a `+experiment=...` pattern expands to must be real.

        Handles shell-style brace groups, including more than one per pattern
        (`model_size/{dino,dinov2}_{small,base}` is nine configs, not two).
        """
        text = README.read_text()
        missing = []
        for pattern in re.findall(r"\+experiment=([\w/,{}-]+)", text):
            for name in _brace_expand(pattern):
                if not (CONFIGS / "experiment" / f"{name}.yaml").is_file():
                    missing.append(name)
        assert missing == [], (
            f"README references non-existent experiment configs: {sorted(set(missing))}"
        )


def _brace_expand(pattern: str) -> list[str]:
    """Expand `a/{x,y}_{1,2}` into every concrete name, like a shell would."""
    match = re.search(r"\{([^{}]*)\}", pattern)
    if not match:
        return [pattern]
    head, tail = pattern[: match.start()], pattern[match.end() :]
    out = []
    for option in match.group(1).split(","):
        out.extend(_brace_expand(f"{head}{option.strip()}{tail}"))
    return out


class TestDocumentedPaths:
    def test_documented_config_directories_exist(self):
        """The Configuration tree lists directories under configs/."""
        text = README.read_text()
        block = re.search(r"```\nconfigs/\n(.*?)```", text, re.S)
        assert block, "Configuration section is missing its directory listing"
        for line in block.group(1).splitlines():
            entry = line.strip().split()[0] if line.strip() else ""
            if entry.endswith("/"):
                assert (CONFIGS / entry.rstrip("/")).is_dir(), (
                    f"README documents configs/{entry} which does not exist"
                )

    def test_documented_source_packages_exist(self):
        text = README.read_text()
        block = re.search(r"```\nsrc/scdino/\n(.*?)```", text, re.S)
        assert block, "Project structure section is missing its listing"
        for line in block.group(1).splitlines():
            entry = line.strip().split()[0] if line.strip() else ""
            if entry.endswith("/"):
                name = entry.rstrip("/")
                assert list((REPO / "src" / "scdino").rglob(name)), (
                    f"README documents src/scdino/{name} which does not exist"
                )

    def test_referenced_repo_files_exist(self):
        """Markdown links to in-repo files must resolve."""
        text = README.read_text()
        missing = [
            target
            for target in re.findall(r"\]\(([^)h][^)]*)\)", text)
            if not (REPO / target.split("#")[0]).exists()
        ]
        assert not missing, f"README links to missing files: {missing}"
