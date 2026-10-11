"""The extension loads only when it was built from the sources beside it.

These run without a GPU: the digest is a pure function of the source tree,
and the loader is exercised against stand-in modules, so a regression in
either is caught on every machine rather than only where the kernels run.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

from kernels import extension


@pytest.fixture
def source_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A copy of ``cpp/`` that the digest reads in place of the real one."""
    copy = tmp_path / "cpp"
    shutil.copytree(extension.SOURCE_DIR, copy)
    monkeypatch.setattr(extension, "SOURCE_DIR", copy)
    return copy


@pytest.fixture
def fresh_load() -> Iterator[None]:
    """Run a test against an uncached loader and leave none of its result."""
    extension.load.cache_clear()
    yield
    extension.load.cache_clear()


def _stand_in(build_sha256: str | None) -> ModuleType:
    module = ModuleType("kernels._C")
    module.__file__ = "/stand-in/_C.so"
    if build_sha256 is not None:
        setattr(module, "build_sha256", build_sha256)
    return module


class TestBuildDigest:
    def test_is_stable_over_an_unchanged_tree(self, source_copy: Path) -> None:
        assert extension.build_sha256() == extension.build_sha256()

    @pytest.mark.parametrize(
        "name", ["graph_build.cu", "graph_build.h", "bindings.cpp"]
    )
    def test_changes_with_any_source_byte(self, source_copy: Path, name: str) -> None:
        before = extension.build_sha256()
        path = source_copy / name
        path.write_bytes(path.read_bytes() + b"\n")
        assert extension.build_sha256() != before

    def test_changes_when_a_source_is_added(self, source_copy: Path) -> None:
        before = extension.build_sha256()
        (source_copy / "extra.cu").write_text("// new kernel\n")
        assert extension.build_sha256() != before

    def test_changes_with_the_compile_flags(
        self, source_copy: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        before = extension.build_sha256()
        monkeypatch.setattr(
            extension, "NVCC_FLAGS", (*extension.NVCC_FLAGS, "--use_fast_math")
        )
        assert extension.build_sha256() != before

    def test_ignores_files_the_build_does_not_read(self, source_copy: Path) -> None:
        before = extension.build_sha256()
        (source_copy / "notes.txt").write_text("not compiled\n")
        assert extension.build_sha256() == before

    def test_builds_every_translation_unit(self) -> None:
        names = {path.name for path in extension.translation_units()}
        assert names == {
            "bindings.cpp",
            "fused_edge_update.cu",
            "fused_norm_residual.cu",
            "graph_build.cu",
        }


@pytest.mark.usefixtures("fresh_load")
class TestLoad:
    def test_refuses_a_build_of_other_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "kernels._C", _stand_in("0" * 64))
        with pytest.raises(ImportError, match="built from other sources or flags"):
            extension.load()

    def test_refuses_an_unstamped_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "kernels._C", _stand_in(None))
        with pytest.raises(ImportError, match="build None"):
            extension.load()

    def test_names_the_build_command_when_nothing_loads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "kernels._C", None)
        with pytest.raises(ImportError, match=extension.BUILD_COMMAND):
            extension.load()

    def test_accepts_a_build_of_the_current_tree(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stand_in = _stand_in(extension.build_sha256())
        monkeypatch.setitem(sys.modules, "kernels._C", stand_in)
        assert extension.load() is stand_in
