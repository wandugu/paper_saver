# Changes from source

The imported revision is recorded in [SOURCE_COMMIT.txt](SOURCE_COMMIT.txt).
Deployment paths and host identifiers were replaced with configurable values.
The seed lists, thresholds, scientific decision rules, and resource hashes
are preserved. Public registration names and IDs are neutralized.

## CPU tests before and after path cleanup

Both runs used the same shared Windows Python 3.13.15, torch 2.14.0+cu130,
and transformers 5.22.2 environment. From the `reproduction/` directory,
the test arguments were `python -m pytest repro/p24/tests -q -rA -p no:cacheprovider --continue-on-collection-errors`.
Each run used a separate `--basetemp` directory outside the checkout and
disabled bytecode writes.

| Test file | Before (pass / fail / skip / collection error) | After (pass / fail / skip / collection error) |
| --- | --- | --- |
| `test_batch_serving.py` | 0 / 0 / 0 / 1 | 0 / 0 / 0 / 1 |
| `test_calibration.py` | 37 / 0 / 0 / 0 | 37 / 0 / 0 / 0 |
| `test_checkpoint_selection.py` | 4 / 0 / 0 / 0 | 4 / 0 / 0 / 0 |
| `test_data.py` | 13 / 1 / 0 / 0 | 13 / 1 / 0 / 0 |
| `test_encoders.py` | 15 / 2 / 0 / 0 | 15 / 2 / 0 / 0 |
| `test_experiment.py` | 22 / 0 / 0 / 0 | 22 / 0 / 0 / 0 |
| `test_gpu_guard.py` | 4 / 0 / 0 / 0 | 4 / 0 / 0 / 0 |
| `test_model.py` | 44 / 0 / 0 / 0 | 44 / 0 / 0 / 0 |
| `test_phase2_audit.py` | 5 / 0 / 0 / 0 | 5 / 0 / 0 / 0 |
| `test_phase2_queue.py` | 0 / 0 / 0 / 1 | 0 / 0 / 0 / 1 |
| `test_phase2_training.py` | 44 / 0 / 0 / 0 | 44 / 0 / 0 / 0 |
| `test_phase3_replay.py` | 8 / 0 / 0 / 0 | 8 / 0 / 0 / 0 |
| `test_protocol.py` | 3 / 0 / 0 / 0 | 3 / 0 / 0 / 0 |
| `test_utility_agreement.py` | 2 / 0 / 0 / 0 | 2 / 0 / 0 / 0 |
| **Total** | **201 / 3 / 0 / 2** | **201 / 3 / 0 / 2** |

The same tests failed in both runs. Windows lacked the POSIX `fcntl`
module in two test files and produced backslashes where one data test
expected slash-separated paths. Two encoder tests failed with
`ModernBertConfig.reference_compile` missing under the shared transformers
version. The collection and test outcomes were identical before and after
cleanup.

## File changes

Paths are relative to the repository root.

- `README.md` — Correct the public record, label unexecuted configurations and commands, and cite logged MNRE results.
- `reproduction/CHANGES_FROM_SOURCE.md` — Record the before/after CPU test comparison and every file changed for publication.
- `reproduction/README.md` — Document P24 scope, source revision, requirements, environment, launch commands and table generation.
- `reproduction/repro/README.md` — Replace the copied monorepo guide with a pointer to the P24 guide.
- `reproduction/repro/SCALED_PROTOCOL.md` — Replace the monorepo protocol with the scope of the P24 experiments.
- `reproduction/repro/common/paths.py` — Share explicit root expansion across the configuration loader and historical input readers.
- `reproduction/repro/p24/README.md` — Replace host-bound instructions with a P24 runner index and registration guidance.
- `reproduction/repro/p24/configs/mre_always_vision_s42.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_always_vision_s43.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_always_vision_s44.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_maxsim_s42.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_maxsim_s43.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_maxsim_s44.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_mean_s42.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_mean_s43.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_mean_s44.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_random_s42.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_random_s43.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_random_s44.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_saver_s42.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_saver_s43.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_saver_s44.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_text_only_s42.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_text_only_s43.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_text_only_s44.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_topk2_s42.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_topk2_s43.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/mre_topk2_s44.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/configs/pilot_local.yaml` — Use MODEL_ROOT for frozen encoder weights.
- `reproduction/repro/p24/d0_sources.md` — Remove internal manuscript and filesystem references from the dataset availability record.
- `reproduction/repro/p24/experiment.py` — Parameterize the designated execution hosts and metadata label while retaining the device and resource checks.
- `reproduction/repro/p24/phase2_plan.py` — Record the configured formal host label in queue plans.
- `reproduction/repro/p24/phase2_queue.py` — Parameterize host and session names while preserving lane and dependency guards.
- `reproduction/repro/p24/phase3_deliver.py` — Use scope, findings, and limitations in the generated report.
- `reproduction/repro/p24/phase3_export.py` — Resolve source, data and phase ledger defaults from the configured roots.
- `reproduction/repro/p24/phase3_replay.py` — Resolve source and receipt path placeholders when replay and audit scripts load JSON.
- `reproduction/repro/p24/phase3_tables.py` — Use the configurable phase root and preserve the existing table-generator override.
- `reproduction/repro/p24/phase5_audit.py` — Parameterize historical runtime, GPU and ledger identity checks.
- `reproduction/repro/p24/phase5_c1_config.json` — Use P24_RUN_ROOT for the existing pilot and anchor records.
- `reproduction/repro/p24/phase5_cpu_queue.py` — Build the CPU queue ledger location from P24_RUN_ROOT.
- `reproduction/repro/p24/phase5_delta1.json` — Use P24_RUN_ROOT for recorded bootstrap inputs while preserving their hashes.
- `reproduction/repro/p24/phase5_pinned_setup.py` — Build the pinned-runtime ledger location from P24_RUN_ROOT.
- `reproduction/repro/p24/phase5_prepare.py` — Build the replay configuration ledger location from P24_RUN_ROOT.
- `reproduction/repro/p24/phase5_preregister.py` — Use the shared path-aware replay reader for source configurations.
- `reproduction/repro/p24/phase5_runtime_exclusions.json` — Use P24_REPO_ROOT for the excluded cache entries while preserving their identities.
- `reproduction/repro/p24/phase5_sources.json` — Parameterize the code, prepared data and ledger locations used by replay.
- `reproduction/repro/p24/phase5_tables.py` — Use the configurable phase root and preserve the existing table-generator override.
- `reproduction/repro/p24/phase5_training.py` — Parameterize the formal hostname and resolve excluded cache paths before exact matching.
- `reproduction/repro/p24/prereg_p24.yaml` — Parameterize deployment paths, remove operational notes, and retain scientific rules.
- `reproduction/repro/p24/prereg_p24_phase2.yaml` — Parameterize deployment paths, remove operational notes, and retain scientific rules.
- `reproduction/repro/p24/prereg_p24_phase3.yaml` — Parameterize deployment paths, remove operational notes, and retain scientific rules.
- `reproduction/repro/p24/prereg_p24_phase4.yaml` — Parameterize deployment paths, remove operational notes, and retain scientific rules.
- `reproduction/repro/p24/prereg_p24_phase5.yaml` — Parameterize deployment paths, remove operational notes, and retain scientific rules.
- `reproduction/repro/p24/profile_run.py` — Parameterize the profiling host, GPU UUID and metadata label while retaining the same hardware checks.
- `reproduction/repro/p24/resources.lock.json` — Remove the source machine’s checkout name while preserving the resource identity hashes.
- `reproduction/repro/p24/resources.py` — Default model acquisition to MODEL_ROOT or a local models directory.
- `reproduction/repro/p24/tests/test_gpu_guard.py` — Use neutral test hosts and exercise the same guard branches.
- `reproduction/repro/requirements-repro310.txt` — List the P24 dependencies for the pinned runtime.
- `reproduction/repro/run.py` — Resolve path placeholders before passing the configuration to the ledger and runner.
- `reproduction/results/tables/p24_mre_main.md` — Copy the unchanged generated Markdown table cited in the top-level README.
- `reproduction/results/tables/p24_phase3_R2_test.md` — Copy the unchanged generated Markdown table cited in the top-level README.
- `reproduction/results/tables/p24_phase5_A6_paired_s42.md` — Copy the unchanged generated Markdown table cited in the top-level README.
- `reproduction/results/tables/p24_phase5_A6_paired_s43.md` — Copy the unchanged generated Markdown table cited in the top-level README.
- `reproduction/results/tables/p24_phase5_A6_paired_s44.md` — Copy the unchanged generated Markdown table cited in the top-level README.
- `reproduction/results/tables/p24_phase5_C3.md` — Copy the unchanged generated Markdown table cited in the top-level README.

## Phase 6 import

[SOURCE_COMMIT.txt](SOURCE_COMMIT.txt) records both the phase 2–5 base and
the phase 6 additions. The test lives in `reproduction/tests/`, matching its
location beside `repro/` in the source checkout. Running `python -m pytest`
from `reproduction/` resolves its existing `repro` imports.

The same shared Windows Python, torch, and transformers environment was used
for the phase 6 verification. The command added `tests/test_p24_phase6_e2.py`
to the existing `repro/p24/tests` test paths with the same pytest options.

| Test files | Pass | Fail | Skip | Errors |
| --- | ---: | ---: | ---: | ---: |
| Original 14 files | 201 | 3 | 0 | 2 |
| `tests/test_p24_phase6_e2.py` | 1 | 0 | 0 | 4 |
| Total | 202 | 3 | 0 | 6 |

The original 14 files match the preceding comparison, including both
collection errors and all three failing test IDs. In the new file,
`test_05_f1_excludes_none` passed. Checks 1–4 reported fixture setup errors
because `runtime(cpu=True)` requires torch 2.6.0+cu118 and transformers
4.57.1. This local rerun stopped those checks before reading the original
artifacts. The historical pinned-runtime validation is described in the
[reproduction guide](README.md#phase-6-status).

### Files added or updated in this import

- `README.md` — Cite the post hoc E1 development-set upper bound, conditional on the fitted models.
- `reproduction/CHANGES_FROM_SOURCE.md` — Record the phase 6 import and test comparison.
- `reproduction/README.md` — Record phase 6 status and document E1 replay and table generation.
- `reproduction/SOURCE_COMMIT.txt` — Record the phase 2–5 base and the combined phase 6 source revision.
- `reproduction/repro/p24/phase6_e1.py` — Resolve registered input paths and identify the public template in newly computed epsilon receipts.
- `reproduction/repro/p24/phase6_e1_config.yaml` — Include the unchanged fixed E1 configuration and registration ID.
- `reproduction/repro/p24/phase6_e1_tables.py` — Use `P24_RUN_ROOT` for the default ledger location and preserve the `P24_PHASE6_RUNS` override.
- `reproduction/repro/p24/phase6_e2.py` — Resolve E2 data, cache and weight locations from the shared path placeholders.
- `reproduction/repro/p24/phase6_epsilon.json` — Include the unchanged historical E1 epsilon receipt.
- `reproduction/repro/p24/phase6_readonly.py` — Include the unchanged cache reader used by E2.
- `reproduction/repro/p24/phase6_statistics.py` — Include the unchanged E1 paired statistics code.
- `reproduction/repro/p24/prereg_p24_phase6_e1.yaml` — Parameterize the pinned executable and phase 5 ledger paths while preserving the registered rules.
- `reproduction/results/tables/p24_phase6_E1a_dev.md` — Include the unchanged generated table cited in the top-level README.
- `reproduction/tests/test_p24_phase6_e2.py` — Include the five source checks and resolve the pinned phase 5 ledger path.

## Publication review changes

- Removed the unrelated compatibility module for a card-game dependency from `reproduction/repro/common/`.
- Removed `reproduction/repro/p24/phase5_directions.md`, the internal planning document.
- Renamed the six venue-labeled registrations to `prereg_p24.yaml`, `prereg_p24_phase2.yaml`, `prereg_p24_phase3.yaml`, `prereg_p24_phase4.yaml`, `prereg_p24_phase5.yaml`, and `prereg_p24_phase6_e1.yaml` in `repro/p24/`. These are public templates. Their IDs, paths, and operational notes differ from the historical originals.
- Updated the registration consumers in `queue.py`, `phase2_plan.py`, `phase2_queue.py`, `phase3_audit.py`, `phase3_export.py`, `phase5_cpu_queue.py`, `phase5_sources.json`, `phase6_e1.py`, `phase6_e1_config.yaml`, `phase6_e2.py`, and `tables.py` under `repro/p24/`. The table spec can read a private historical-ID map. Phase 2 plans take the run deadline explicitly.
- Updated `repro/common/envinfo.py` and `repro/requirements-repro310.txt` to cover only the P24 workflow. The receipt's top-level keys retain their format. Its `versions` mapping now reports torch, transformers, and NumPy.
- Updated `repro/common/guard.py` with default and configurable blocking rules for imports and file opens. `repro/common/paths.py` and `repro/common/prereg.py` use the same strict substitution as the run-config loader.
- Updated `repro/p24/d0_sources.md` to retain public dataset citations and `phase3_deliver.py` / `phase3_verify.py` to describe scope, findings, and limitations.
- Updated `repro/p24/experiment.py`, `profile_run.py`, `batch_serving.py`, `phase2_queue.py`, `phase5_training.py`, and `phase5_audit.py` to require configured hosts, session prefixes, or device identifiers where checked. The historical audit compares executable strings exactly. The queue and profiling modules import their POSIX lock support where they acquire the lock.
- Updated `repro/p24/phase5_runtime_exclusions.json`, `phase5_training.py`, `phase5_anchor.py`, `phase5_audit.py`, `phase6_readonly.py`, and `phase6_e2.py` to apply exclusions relative to the cache reader's feature root.
- Updated `repro/p24/phase5_analysis.py` and `phase6_e1.py` to map historical E1b input locations in memory. The bundled historical epsilon receipt keeps its original digest. For a new calculation, the receipt hashes the registration file supplied for that run.
- Updated `repro/p24/resources.py` to require the revisions and weight hashes from its lock when downloading resources.
- Updated the root `README.md`, `reproduction/README.md`, and the runner index for the evidence scope, setup, registration provenance, phase 6 status, and license.
- Updated `repro/p24/tests/test_batch_serving.py`, `test_gpu_guard.py`, `test_phase2_queue.py`, and `tests/test_p24_phase6_e2.py` for the runtime changes. Added `tests/test_release_integrity.py` for import and open interception, cache relocation, preregistration expansion, the resource lock, and historical path mapping.

The review rerun in the shared Windows environment recorded 258 passes,
3 failures, no skips, and 4 setup errors. Windows can now collect the two
queue and profiling test files by deferring the POSIX lock imports. The
remaining data-path and encoder failures and the four phase 6 version
guard errors are the same as in the prior run.

## Follow-up publication review

- Restored the original phase 5 A4 reporting clause and A3b, A7 and A9
  launch conditions. The C3 no-go receipt explicitly stops running A7
  work and prevents further A7 launches. The six-template comparison lists
  every remaining YAML field difference and its reason in
  [PREREG_TEMPLATE_DIFF.md](PREREG_TEMPLATE_DIFF.md).
- The phase 3 report uses pytest execution summaries and logged metrics.
  The exporter records historical registration commits separately from the
  hashes of copied public templates.
- Phase 2 agreement jobs use their plan's time-zone-aware deadline. The
  queue stops when tmux cannot confirm the queried session state.
- The runner hashes the same registration bytes it parses. New runs record
  the digest in their receipt and ledger. The phase 3 method lock and E1
  epsilon calculation use those run records, and fixed templates are checked
  before execution.

The follow-up added 23 passing cases. The shared Windows rerun recorded
281 passes, 3 failures, no skips, and 4 setup errors. The same three tests
and four E2 fixtures account for all failures and errors.
