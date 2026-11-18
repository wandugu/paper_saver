"""Read exactly the cache namespace used by the pinned phase5 controls."""

import hashlib
import json
from pathlib import Path

import torch

from .phase5_training import cache_path
from .protocol import FeatureStore, cache_key


class CacheMiss(FileNotFoundError):
    pass


class ReadOnlyStore(FeatureStore):
    def __init__(self, root, encoders, identity):
        namespace = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        self.root = Path(root) / namespace
        self.encoders = encoders
        self.misses = []
        if not self.root.is_dir():
            self.miss("namespace", self.root)

    def miss(self, kind, path, key=None):
        self.misses.append({"kind": kind, "path": str(path), "key": key})
        raise CacheMiss(f"{kind} cache miss for {key}: {path}")

    def create(self, path, value):
        raise PermissionError(f"Phase6 feature cache is read-only: {path}")

    def load(self, path, kind, key):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError:
            self.miss(kind, path, key)

    def text_global(self, record, task, visual=True):
        path = cache_path(self.root.parent, self.root / (cache_key(record) + ".pt"))
        payload = self.load(path, "text_global", record["id"])
        if visual and payload.get("global") is None:
            self.miss("global", path, record["id"])
        return payload

    def detail(self, image, cache=True):
        key = hashlib.sha256(str(image).encode()).hexdigest()
        path = cache_path(self.root.parent, self.root / ("detail-" + key + ".pt"))
        return self.load(path, "detail", str(image)).to(self.encoders.device)
