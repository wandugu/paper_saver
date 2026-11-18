# P24 scaled protocol

The logged task is multimodal relation extraction on MNRE. It uses frozen
ModernBERT and SigLIP 2 encoders, a single NVIDIA T4 per GPU run, and the
official dataset splits. The runs include the learned SAVER router, an
independently trained text-only model, always-vision and matched routing
controls.

The phase-specific preregistrations and runner configurations are in
[`p24/`](p24/). Use the registration for the phase being reproduced. The
later phases analyze text fallback, calibration against an independent text
anchor, and the effect of removing the training and development caps.
Development and calibration determine choices within those phases. The
later analyses followed prior inspection of test results and use test
metrics descriptively.

See the [reproduction guide](../README.md) for data and ledger inputs.
