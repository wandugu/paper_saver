"""Materialize fixed CPU replay configs and cache job config."""

import json
import os
from pathlib import Path


def main():
    runs = Path(os.environ.get("P24_RUN_ROOT", "runs")) / "p24_phase5"
    destination = Path("tmp/p24_phase5/configs")
    for seed in (42, 43, 44):
        for scope in ("reduced", "full"):
            config = {"sources": str(Path("repro/p24/phase5_sources.json").resolve()), "scope": scope}
            with (destination / f"anchor_{scope}_s{seed}.json").open("x") as stream:
                json.dump(config, stream, indent=2)
        for study in ("A1", "A2", "legacy"):
            config = {"sources": str(Path("repro/p24/phase5_sources.json").resolve()), "study": study,
                      "phase5_root": str(runs)}
            with (destination / f"analysis_{study}_s{seed}.json").open("x") as stream:
                json.dump(config, stream, indent=2)
    config = json.loads((destination / "A1_s42.json").read_text())
    with (destination / "cache.json").open("x") as stream:
        json.dump(config, stream, indent=2)


if __name__ == "__main__":
    main()
