# P24 SAVER reproduction

This harness implements the scaled SAVER experiments and baselines for MNRE
multimodal relation extraction. The logged GPU runs used frozen ModernBERT
and SigLIP 2 encoders, with one NVIDIA T4 per run. The package includes the
original reduced protocol, the preregistrations for phases 2–5 and phase 6 E1,
and calibration, prediction replay, audit, and ledger-based table scripts. See the
[runner index](repro/p24/README.md) for the phase entries.

The phase 2–5 code and phase 6 additions come from the private research
repository revisions recorded in [`SOURCE_COMMIT.txt`](SOURCE_COMMIT.txt).
The combined P24 source tree corresponds to the phase 6 revision. This
directory is released under the [repository license](../LICENSE).

The published preregistration files are public templates with parameterized paths, operational notes removed, and neutral IDs. Historical runs used earlier IDs and the original files, whose hashes are kept with the private run records. The later phases were registered before their new runs and after earlier test results had been seen. They use development and calibration data for decisions and report test results descriptively. See [the change and test record](CHANGES_FROM_SOURCE.md).

The runner parses and hashes the same registration byte snapshot, before
environment expansion, and records the SHA-256 in `preregistration.json` and
`ledger.jsonl`. Phase 2 jobs check the hash recorded by their plan. The
phase 3 audit and phase 6 fixed-template entry points require the digest
and resolved fields to match their bundled registration. The phase 3 method
lock records the run's digest. The E1 epsilon calculation uses the run's
receipt. For a historical run without that receipt, pass the original
registration as `prereg_path` to `phase6_e1.calculate_epsilon` and use its
original bytes to verify the historical digest, including the digest in
the bundled `phase6_epsilon.json`. See the
[registration comparison](PREREG_TEMPLATE_DIFF.md) for public-template changes.

Historical runs used an additional private blocklist for modules absent
from this repository, so the default rules cover the published code.

## Phase 6 status

E1 completed as a post hoc, CPU-only paired replay of saved phase 5 runs.
It used existing predictions and performed no new training or encoder calls.
E2 is preparation for a text fine-tuning control. No E2 training run took
place. The original CPU validation stopped at check 2 when the CPU fp32
word states failed comparison with the fp16 cache on 3 of 64 records.
The original validation ended before checks 3–5, and E2 was never preregistered.

## Requirements and inputs

Use Python 3.10. For GPU runners, use Linux with an NVIDIA GPU, a
CUDA-compatible driver, and `nvidia-smi`. The queue and profiling modules
use POSIX file locking. The P24 requirements include torch 2.6.0+cu118
and transformers 4.57.1 for the pinned phase 5 and 6 runtime.

```bash
python -m pip install -r repro/requirements-repro310.txt
```

New environment receipts report `torch`, `transformers`, and `numpy` in
their `versions` mapping. The top-level receipt fields keep their format.

The checkout includes the code, registration files, and selected generated
Markdown tables. Supply the original MNRE annotations and images, the
resource inventory, and the frozen weights separately. Historical replay,
audits, and table generation additionally require the corresponding ledgers
and run artifacts. Ancestry checks also require the original Git objects.

Run the commands below from this directory. `data` prepares the official
MNRE splits from the release directory containing `txt/`, `img_org/`, and
`ours_rel2id.json`. `resources` requests the exact encoder revisions in
`resources.lock.json` and verifies the returned revision and weight hash.

```bash
export P24_REPO_ROOT="$PWD"
export DATASET_ROOT="$PWD/data"
export MODEL_ROOT="/path/to/local/encoder-weights"
export P24_RUN_ROOT="$PWD/runs"

python -m repro.p24.data mnre --root "$DATASET_ROOT/RE_data" \
  --output "$DATASET_ROOT/p24/mnre"
python -m repro.p24.resources --models-root "$MODEL_ROOT" \
  --inventory "$DATASET_ROOT/p24/resource_inventory.json"
```

The supplied training configurations expect manifests and inventory under
`data/p24/` relative to this directory. Edit the input paths in a copy of the
configuration for a different layout. The data command writes `mnre.jsonl`;
the formal configs refer to the historical `mnre_remote.jsonl`. The resource
lock checks the recorded manifest and weight identities. Regenerating the
manifest with different absolute image paths changes its hash.

## Environment variables

| Variable | Purpose |
| --- | --- |
| `DATASET_ROOT` | Prepared data root with a `p24/` subdirectory for replay sources, export, and the E2 feature cache. |
| `MODEL_ROOT` | Parent of the `ModernBERT-base` and `siglip2-base-patch16-naflex` weight directories. |
| `P24_REPO_ROOT` | Root containing `repro/` and the default early phase ledger directories. |
| `P24_RUN_ROOT` | Parent for the later phase-specific ledger roots and output directories used in the examples. |
| `REPRO_RUNS_ROOT` | Runner default when `--runs-root` is omitted. |
| `P24_FORMAL_HOSTNAME` and `P24_PILOT_HOSTNAME` | Exact unqualified hostnames for GPU checks. Set both for formal phase 1 training. The formal hostname also selects the pinned formal MNRE manifest hash for resource validation. |
| `P24_FORMAL_HOST` | Formal host label recorded in profiling and queue metadata. |
| `P24_PROFILE_GPU_UUID` and `P24_PHASE5_GPU_UUID` | Designated profiling GPU and phase-five historical audit GPU identities. |
| `P24_TMUX_PREFIX` | Required exact prefix of the phase 1 and phase 2 session names checked by the phase 2 queue. |
| `CUDA_VISIBLE_DEVICES` | A single physical index for GPU jobs, or an empty value for CPU-only runs. Phase-five GPU jobs require index `1`. |
| `HF_ENDPOINT`, `HTTPS_PROXY`, and `https_proxy` | Optional endpoint and proxy overrides for the resource downloader. |
| `P24_PHASE1_RUNS_ROOT`, `P24_PHASE2_RUNS_ROOT` | Optional early ledger directories for the phase 3 export. The phase 1 override is also used by E1b replay. |
| `P24_PHASE3_RUNS_ROOT`, `P24_PHASE5_RUNS`, and `P24_PHASE6_RUNS` | Ledger overrides for table specs that read a ledger when imported. |
| `P24_PREREG_ID_MAP` | Optional path to a private JSON mapping from public registration IDs to historical IDs for the phase 1 and 2 table spec. Fresh runs use the public IDs by default. |
| `P24_PHASE6_TABLES` | Output directory for the E1 table spec. Use a fresh directory for each generation. |
| `REPRO_BLOCK_MODULES`, `REPRO_BLOCK_PATHS` | Optional semicolon-separated glob patterns for additional blocked modules and paths. Set before importing `repro`. The default rules block trace-store module and path components, and loading fails if blocked code was already imported. |

## Launch a run

The runner imports `repro.p24.<entry>`, reads the YAML or JSON passed to
`--config`, and writes a ledger row plus run artifacts under `--runs-root`.
Use the registration for the corresponding phase with `--prereg`. For
example, with the locked formal inputs in place and a designated idle GPU
on the configured formal host:

```bash
export P24_FORMAL_HOSTNAME="$(hostname -s)"
export P24_FORMAL_HOST="$P24_FORMAL_HOSTNAME"
: "${P24_PILOT_HOSTNAME:?Set this to the designated pilot hostname}"
export P24_PILOT_HOSTNAME
export CUDA_VISIBLE_DEVICES=0
python -m repro.run p24 experiment \
  --config repro/p24/configs/mre_saver_s42.yaml --seed 42 \
  --prereg repro/p24/prereg_p24.yaml --runs-root "$P24_REPO_ROOT/tmp/p24_phase1/runs"
```

For CPU replay entries such as `phase3_audit` or `phase5_analysis`, pass a
config that points to the relevant source runs. To prepare a new phase 2
queue plan, pass a future ISO 8601 timestamp with a time zone to
`python -m repro.p24.phase2_plan --deadline` and set `P24_TMUX_PREFIX`
to match the exact session names.
The three agreement jobs use the plan's deadline. Legacy agreement
configs with only `source_runs_root` and `teacher` retain the original
deadline of `2026-11-15T23:59:59+08:00` when replayed with their original
registration.

With the locked phase 5 artifacts and pinned runtime available, the E1
replay uses this command.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 nice -n 19 python -m repro.run p24 phase6_e1 \
  --config repro/p24/phase6_e1_config.yaml --seed 0 \
  --prereg repro/p24/prereg_p24_phase6_e1.yaml \
  --runs-root "$P24_RUN_ROOT/p24_phase6"
```

For historical E1b replay, keep the archived inputs read-only and write new
outputs under a separate `P24_RUN_ROOT`. On Linux, expose the pinned phase 5
ledger there as `p24_phase5_pinned` and set the phase 1 ledger override.

```bash
mkdir -p "$P24_RUN_ROOT"
ln -s /path/to/read-only/phase5-ledger "$P24_RUN_ROOT/p24_phase5_pinned"
export P24_PHASE1_RUNS_ROOT=/path/to/read-only/phase1-ledger
export DATASET_ROOT=/path/to/read-only/prepared-data
```

Each ledger directory must contain its original `ledger.jsonl` and the run
directories it references. Use the original normalized MNRE manifest under
`$DATASET_ROOT/p24/`. E1b uses the mapped source locations in memory and
leaves the archived configs and receipts intact. The existing input hash
and exact replay checks still apply.

## Build tables from run artifacts

Pass a directory containing `ledger.jsonl` and its referenced run directories,
a Python spec defining `TABLES`, and an output directory.

```bash
export P24_PHASE5_RUNS="$P24_RUN_ROOT/p24_phase5_pinned"
python -m repro.make_tables --runs-root "$P24_PHASE5_RUNS" \
  --spec repro/p24/phase5_tables.py --out-dir "$P24_RUN_ROOT/tables/phase5"
```

Use `repro/p24/tables.py` for the original and phase-two tables,
`repro/p24/phase3_tables.py` for phase three, and
`repro/p24/phase4_tables.py` for phase four. For the phase-five tables linked
from the top-level README, use the pinned-runtime ledger. The earlier
phase-five attempts from the alternate runtime are excluded by the
registration. The published Markdown copies are in
[`results/tables/`](results/tables/).

To render phase 1 or 2 tables from historical ledgers, put the exact IDs
from those ledgers in a private JSON object keyed by `P24-phase1-v1` and
`P24-phase2-v1`. Point `P24_PREREG_ID_MAP` at that file for this invocation.
Use a separate invocation without the mapping to render public-template runs.

To generate the E1 tables from its completed run, point the spec at its
ledger and a fresh output directory.

```bash
export P24_PHASE6_RUNS="$P24_RUN_ROOT/p24_phase6"
export P24_PHASE6_TABLES="$P24_RUN_ROOT/tables/phase6"
python -m repro.make_tables --runs-root "$P24_PHASE6_RUNS" \
  --spec repro/p24/phase6_e1_tables.py --out-dir "$P24_PHASE6_TABLES"
```

Run the synthetic and tiny-backbone tests with
`python -m pytest repro/p24/tests tests/test_release_integrity.py -q`.
The E2 CPU checks require the pinned runtime, original MNRE data, encoder
weights, phase 5 ledger, and feature cache. With the pinned formal MNRE
manifest in place on the validation host, run the ordered checks as follows.

```bash
export P24_FORMAL_HOSTNAME="$(hostname -s)"
CUDA_VISIBLE_DEVICES="" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -x -q tests/test_p24_phase6_e2.py
```
