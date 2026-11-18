# P24 runners

See the [reproduction guide](../../README.md) for setup, inputs, and runnable
commands. The YAML files in this directory are the public registration
templates for new runs.

- `experiment` and `pilot` run the reduced frozen-encoder training path.
  `profile_run` measures a locked checkpoint on development posts.
- `phase2_training`, `phase2_audit`, `checkpoint_selection`,
  `utility_agreement`, and `batch_serving` cover router and teacher comparisons,
  post-level calibration, checkpoint selection, agreement, and batched
  serving.
- `phase3_extract`, `phase3_audit`, and `phase3_verify` examine locked policy
  replay, calibration sensitivity, and split and timing provenance.
  `phase4_extract` and `phase4_audit` support the cost and harm certificate
  replay and matched controls.
- `phase5_training` tests repaired text fallback and the uncapped
  text-only and always-vision controls. `phase5_checkpoint` evaluates the
  registered gates, and `phase5_analysis`, `phase5_diagnostics`,
  `phase5_replay`, and `phase5_audit` analyze and check saved runs.
- `phase6_e1` replays saved phase 5 predictions on CPU. `phase6_e2` is a
  draft text fine-tuning control. Its original CPU validation stopped at
  check 2, before preregistration and training.

The resource lock records the dataset manifest and encoder weight hashes.
The calibration code records all-off outcomes and the tables preserve
missing entries. See the phase registrations for the design and gates.
