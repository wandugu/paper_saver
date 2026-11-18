"""Create distinct pinned-runtime configs, preserving all excluded runs."""

import json
import os
from pathlib import Path


def main():
    original = Path("tmp/p24_phase5/configs")
    target = Path("tmp/p24_phase5_pinned/configs")
    target.mkdir(parents=True, exist_ok=False)
    runs = str(Path(os.environ.get("P24_RUN_ROOT", "runs")) / "p24_phase5_pinned")
    for path in original.glob("*.json"):
        config = json.loads(path.read_text())
        if "phase5_root" in config:
            config["phase5_root"] = runs
        if "config_dir" in config:
            config["config_dir"] = str(target.resolve())
        if "checkpoint" in config:
            continue
        with (target / path.name).open("x") as stream:
            json.dump(config, stream, indent=2)


if __name__ == "__main__":
    main()
