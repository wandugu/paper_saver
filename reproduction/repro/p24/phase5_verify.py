"""CPU checks for anchor identity, frozen parameters and uncapped splits."""

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from .encoders import FrozenEncoders
from .experiment import _forward, verify_resources
from .model import SaverCore
from .phase3_replay import calibration_populations
from .phase5_training import AppendOnlyStore, FROZEN_MODULES, build_model
from .protocol import read_records, split_records


def main(ctx, config):
    if torch.cuda.is_available():
        raise RuntimeError("Verification is CPU-only")
    torch.set_num_threads(1)
    ctx.tags.update(stage="phase5", study=config.get("study", "implementation_checks"), device="cpu")
    results = {}
    for seed in config.get("seeds", (42, 43, 44)):
        settings_path = config.get("settings_by_seed", {}).get(str(seed), str(Path(config["config_dir"]) / f"A2_s{seed}.json"))
        settings = json.loads(Path(settings_path).read_text())
        encoders = SimpleNamespace(device=torch.device("cpu"), span_features=FrozenEncoders.span_features)
        identity = verify_resources(settings, "mre")
        store = AppendOnlyStore(settings["cache"], encoders, identity)
        model, loading = build_model(settings, seed)
        reference = SaverCore(settings["num_labels"])
        reference.load_state_dict(torch.load(settings["anchor_checkpoint"], map_location="cpu", weights_only=True))
        model.eval()
        reference.eval()
        source_identity = json.loads((Path(settings["anchor_checkpoint"]).parent / "data_identity.json").read_text())
        records = read_records(settings["manifest"])
        splits = split_records(records, seed, fit_cap=settings.get("fit_cap", 4096), dev_cap=settings.get("dev_cap", 512))
        checked = 0
        with torch.inference_mode():
            for record in splits["dev"]:
                anchor, _, _ = _forward(reference, store, record, "mre", source_identity["labels"], "text_only", 1000)
                routed, _, _ = _forward(model, store, record, "mre", source_identity["labels"], "saver", 1000,
                                        threshold=float("inf"), force_details=True)
                if not torch.equal(anchor["logits"], routed["logits"]):
                    raise AssertionError(f"Bitwise anchor identity failed: seed{seed}/{record['id']}")
                checked += 1
        if checked != config.get("expected_dev_units", 512) or store.new_bytes:
            raise AssertionError("Incomplete dev identity check or cache mutation")
        if any(parameter.requires_grad for name, parameter in model.named_parameters() if name.split(".")[0] in FROZEN_MODULES):
            raise AssertionError("Anchor parameter remains trainable")
        output, _, _ = _forward(model, store, splits["dev"][0], "mre", source_identity["labels"], "saver", 1000, training=True)
        output["loss"].backward()
        if any(parameter.grad is not None for name, parameter in model.named_parameters() if name.split(".")[0] in FROZEN_MODULES):
            raise AssertionError("Frozen anchor received gradient")
        uncapped = split_records(records, seed, fit_cap=None, dev_cap=None)
        full_identity = {"split_ids": {name: [row["id"] for row in rows] for name, rows in uncapped.items()}}
        populations = calibration_populations(records, full_identity, seed, fractions=(.02, .05, .1))
        if set(row["id"] for row in populations[.1]) != set(full_identity["split_ids"]["calibration"]):
            raise AssertionError("Uncapped calibration mismatch")
        try:
            calibration_populations(records, full_identity, seed)
        except ValueError as error:
            if "Insufficient unused" not in str(error):
                raise
        else:
            raise AssertionError("Uncapped 20% must not be synthesized")
        results[str(seed)] = {"bitwise_equal_units": checked, "fit_units_uncapped": len(uncapped["fit"]),
                              "dev_units_uncapped": len(uncapped["dev"]), "calibration_units": len(uncapped["calibration"]),
                              "loading": loading, "anchor_gradients_absent": True, "calibration_option_b": True}
    ctx.save_json("verification.json", results)
    ctx.set_metrics({"all_checks_pass": True, "bitwise_equal_units": sum(row["bitwise_equal_units"] for row in results.values()), "new_cache_bytes": 0})
