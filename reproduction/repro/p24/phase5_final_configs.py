"""Make the fixed extension-complete CPU analysis configs."""

import json
from pathlib import Path


def main():
    root = Path("tmp/p24_phase5/configs")
    for study, seeds in (("A1", (42,)), ("A2", (42, 43, 44))):
        for seed in seeds:
            config = json.loads((root / f"analysis_{study}_s{seed}.json").read_text())
            config.update(include_extension=True, analysis_revision="v4")
            with (root / f"analysis_full_{study}_s{seed}.json").open("x") as stream:
                json.dump(config, stream, indent=2)


if __name__ == "__main__":
    main()
