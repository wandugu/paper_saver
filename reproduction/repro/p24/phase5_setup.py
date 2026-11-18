"""Create explicit phase5 configs from committed source identities."""

import json
from pathlib import Path

from .phase3_replay import json_read
from .phase4_replay import find_run


def main():
    sources = json_read("repro/p24/phase5_sources.json")
    destination = Path("tmp/p24_phase5/configs")
    destination.mkdir(parents=True, exist_ok=True)
    for seed in (42, 43, 44):
        text, _ = find_run(sources["phase1_root"], seed, experiment="experiment", tag_key="method",
                           tag_value="text_only", extra_tags={"stage": "formal"})
        base = json_read(text / "config.json")
        for name in ("cache", "inventory", "manifest"):
            base[name] = str(Path(sources["source_repo"]) / base[name])
        base.update(num_labels=len(json_read(text / "data_identity.json")["labels"]), seed=seed)
        for variant in ("A1", "A2"):
            config = {**base, "method": "saver", "study": variant, "variant": variant,
                      "fit_cap": 4096, "dev_cap": 512, "anchor_checkpoint": str(text / "best.pt")}
            with (destination / f"{variant}_s{seed}.json").open("x") as stream:
                json.dump(config, stream, indent=2)
        for method in ("text_only", "always_vision"):
            config = {**base, "method": method, "study": "A6", "variant": "control", "fit_cap": None, "dev_cap": None}
            with (destination / f"A6_{method}_s{seed}.json").open("x") as stream:
                json.dump(config, stream, indent=2)
    with (destination / "verify.json").open("x") as stream:
        json.dump({**sources, "config_dir": str(destination.resolve())}, stream, indent=2)


if __name__ == "__main__":
    main()
