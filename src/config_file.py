"""Reading a YAML config file into the field values it sets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def read_config_fields(
    path: Path,
    keys: tuple[str, ...],
    sections: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> dict[str, Any]:
    """Read a YAML config into field values, refusing every key it does not read.

    A key no field reads is a setting nobody gets: a misspelling would run at
    the code default, a section of a removed feature would run without it.

    Parameters
    ----------
    path : Path
        YAML file holding one mapping.
    keys : tuple of str
        Top-level keys, each naming a field.
    sections : tuple of (str, tuple of str)
        Nested mappings by name, with the keys each holds; their fields are
        flattened into the result.

    Returns
    -------
    dict
        Field name to raw value, for the fields the file sets.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the document or a section is not a mapping, or the file sets a key
        no field reads; the message names the file, every such key and the
        accepted ones.
    """
    raw = _require_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path)
    section_names = {name for name, _ in sections}
    fields = {key: raw[key] for key in keys if key in raw}
    unread = [str(key) for key in raw if key not in keys and key not in section_names]
    for name, section_keys in sections:
        if name not in raw:
            continue
        nested = _require_mapping(raw[name], path, section=name)
        fields.update({key: nested[key] for key in section_keys if key in nested})
        unread += [f"{name}.{key}" for key in nested if key not in section_keys]
    if unread:
        accepted = [*keys, *(f"{name}.{key}" for name, ks in sections for key in ks)]
        raise ValueError(
            f"Config {path} sets keys no field reads: {sorted(unread)}; "
            f"accepted keys are {sorted(accepted)}"
        )
    return fields


def _require_mapping(
    value: object, path: Path, *, section: str | None = None
) -> dict[Any, Any]:
    """Return *value* if it is a mapping, else raise naming the file."""
    if isinstance(value, dict):
        return value
    where = "the document" if section is None else f"section {section!r}"
    raise ValueError(
        f"Config {path}: {where} must be a mapping, got {type(value).__name__}"
    )
