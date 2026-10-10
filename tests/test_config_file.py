"""A YAML config is read whole or refused."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest

from config_file import read_config_fields
from evaluation.config import EvalConfig
from training.config import TrainConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
SCRIPTS_DIR = REPO_ROOT / "scripts"

ConfigClass = type[TrainConfig] | type[EvalConfig]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestReadConfigFields:
    def test_flattens_sections_into_the_fields_they_hold(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "seed: 7\nmodel:\n  hidden_dim: 32\n")
        fields = read_config_fields(
            path, ("seed", "patience"), (("model", ("hidden_dim", "dropout")),)
        )
        assert fields == {"seed": 7, "hidden_dim": 32}

    def test_every_unread_key_is_named_in_one_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, "seed: 7\nsede: 8\nmodel:\n  hidden: 32\n")
        with pytest.raises(ValueError) as exc_info:
            read_config_fields(path, ("seed",), (("model", ("hidden_dim",)),))
        message = str(exc_info.value)
        assert str(path) in message
        assert "'sede'" in message
        assert "'model.hidden'" in message
        assert "'model.hidden_dim'" in message  # the accepted keys are listed

    @pytest.mark.parametrize("text", ["", "- seed\n- 7\n", "7\n"])
    def test_a_document_that_is_not_a_mapping_is_refused(
        self, text: str, tmp_path: Path
    ) -> None:
        path = _write(tmp_path, text)
        with pytest.raises(ValueError, match="the document must be a mapping"):
            read_config_fields(path, ("seed",))

    @pytest.mark.parametrize("text", ["model: 3\n", "model:\n"])
    def test_a_section_that_is_not_a_mapping_is_refused(
        self, text: str, tmp_path: Path
    ) -> None:
        path = _write(tmp_path, text)
        with pytest.raises(ValueError, match="section 'model' must be a mapping"):
            read_config_fields(path, (), (("model", ("hidden_dim",)),))

    def test_a_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_config_fields(tmp_path / "absent.yaml", ("seed",))


class TestConfigClasses:
    @pytest.mark.parametrize(
        ("config_cls", "path"),
        [
            *((TrainConfig, p) for p in sorted(CONFIGS_DIR.glob("train*.yaml"))),
            *((EvalConfig, p) for p in sorted(CONFIGS_DIR.glob("eval*.yaml"))),
        ],
        ids=lambda value: value.name if isinstance(value, Path) else "",
    )
    def test_every_committed_config_is_read_whole(
        self, config_cls: ConfigClass, path: Path
    ) -> None:
        assert config_cls.from_yaml(path).operation

    @pytest.mark.parametrize("config_cls", [TrainConfig, EvalConfig])
    def test_a_misspelled_key_is_refused(
        self, config_cls: ConfigClass, tmp_path: Path
    ) -> None:
        path = _write(tmp_path, 'operation: "memory"\nsede: 7\n')
        with pytest.raises(ValueError, match="'sede'"):
            config_cls.from_yaml(path)

    def test_a_section_of_a_feature_no_field_reads_is_refused(
        self, tmp_path: Path
    ) -> None:
        """A staged-distance section would otherwise train on every distance."""
        path = _write(
            tmp_path,
            'operation: "memory"\ncurriculum:\n  stages:\n    - distances: [3]\n',
        )
        with pytest.raises(ValueError, match="'curriculum'"):
            TrainConfig.from_yaml(path)

    def test_a_misspelled_section_key_is_refused(self, tmp_path: Path) -> None:
        path = _write(tmp_path, 'operation: "memory"\nmodel:\n  hiden_dim: 32\n')
        with pytest.raises(ValueError, match="'model.hiden_dim'"):
            TrainConfig.from_yaml(path)


class TestScripts:
    @pytest.fixture(scope="class")
    def train_script(self) -> dict[str, Any]:
        return runpy.run_path(str(SCRIPTS_DIR / "train_gnn.py"))

    @pytest.fixture(scope="class")
    def eval_script(self) -> dict[str, Any]:
        return runpy.run_path(str(SCRIPTS_DIR / "eval_gnn.py"))

    def test_training_requires_a_config(self, train_script: dict[str, Any]) -> None:
        with pytest.raises(SystemExit):
            train_script["parse_args"]([])

    @pytest.mark.parametrize("script", ["train_script", "eval_script"])
    def test_a_named_config_that_does_not_exist_raises(
        self, script: str, request: pytest.FixtureRequest, tmp_path: Path
    ) -> None:
        parse_args = request.getfixturevalue(script)["parse_args"]
        with pytest.raises(FileNotFoundError):
            parse_args(["-c", str(tmp_path / "absent.yaml")])
