"""Path resolution helpers for centralized datasets and model weights.

Environment variables are optional so the original project-relative paths remain
valid on machines that do not use the centralized resource layout.
"""

from __future__ import annotations

import os

_DATASET_PREFIXES = (
    ("data/RE_data/", "MNRE/RE_data/"),
    ("data/NER_data/twitter2015/", "MNER/mner2015/twitter2015/"),
    ("data/NER_data/twitter2015_images", "MNER/mner2015/twitter2015_images"),
    ("data/NER_data/twitter2015_aux_images", "MNER/mner2015/twitter2015_aux_images"),
    ("data/NER_data/twitter15_detect", "MNER/mner2015/twitter15_detect"),
    ("data/NER_data/twitter2017/", "MNER/mner2017/twitter2017/"),
    ("data/NER_data/twitter2017_images", "MNER/mner2017/twitter2017_images"),
    ("data/NER_data/twitter2017_aux_images", "MNER/mner2017/twitter2017_aux_images"),
    ("data/NER_data/twitter17_detect", "MNER/mner2017/twitter17_detect"),
)


def _join_root(root: str, relative_path: str) -> str:
    """Join a Windows or POSIX root to slash-separated resource components."""
    return os.path.join(root, *relative_path.strip("/\\").split("/"))


def resolve_dataset_path(path: str) -> str:
    """Resolve a known project dataset path through DATASET_ROOT when set."""
    dataset_root = os.environ.get("DATASET_ROOT")
    if not dataset_root:
        return path
    for old_prefix, central_prefix in _DATASET_PREFIXES:
        if path.startswith(old_prefix):
            suffix = path[len(old_prefix):]
            return _join_root(dataset_root, central_prefix + suffix)
    return path


def resolve_model_path(path: str) -> str:
    """Resolve the legacy local RoBERTa directory through MODEL_ROOT."""
    model_root = os.environ.get("MODEL_ROOT")
    if model_root and path.replace("\\", "/") == "models/roberta-base":
        return _join_root(model_root, "roberta-base")
    return path
