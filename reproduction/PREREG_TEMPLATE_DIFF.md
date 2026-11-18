# Public registration template comparison

The six public YAML templates were compared with the corresponding source exports
by recursively comparing parsed mapping keys, sequence positions and scalar values.
The tables list every differing field. Original private identifiers and paths are
omitted from the comparison text. Formatting and comments are outside the field comparison.

All six `gates` arrays match. In phase 5, the full A4 reporting clause and the A3b,
A7 and A9 conditions match the original. This includes the A7 pilot's concurrency
with A6, the launch cutoff, and the optional work window.

## prereg_p24.yaml

| Field | Change | Justification |
| --- | --- | --- |
| `budgets.feature_preparation_hours_max` | Removed | Omit the historical preparation time allocation. |
| `budgets.forbidden` | Removed | Omit site-specific GPU assignments and repository operation instructions. Runtime safeguards remain in the runner. |
| `budgets.hosts` | Changed | Omit the private placement of training, ledgers and feature preparation. Preserve same-device latency comparisons. |
| `budgets.per_run_hours_max` | Removed | Omit the historical per-run allocation. |
| `budgets.total_gpu_hours_max` | Removed | Omit the historical campaign allocation. |
| `deadline` | Removed | Omit the original wall-clock deadline for the execution plan. |
| `id` | Changed | Use the neutral public registration ID. |
| `models.scope` | Changed | Describe the same frozen-encoder scope independently of a manuscript. |
| `pilot.failure_action` | Changed | Use neutral wording for stopping further runs after a failed pilot. Retain the failure and tuning policy. |
| `profiling.gpu` | Changed | Parameterize the host via `P24_FORMAL_HOST`. Retain the T4 and batch size. |
| `purpose` | Changed | Keep recorded observations as the only inputs, removing the internal manuscript reference. |

## prereg_p24_phase2.yaml

| Field | Change | Justification |
| --- | --- | --- |
| `deadline` | Removed | Omit the original wall-clock deadline for the execution plan. |
| `execution.failures` | Changed | Omit the operational child-process instruction from the template. The queue manages its own child. Preserve how failures are recorded and the restrictions on selection and retries. |
| `execution.host` | Changed | Parameterize the host via `P24_FORMAL_HOST`. |
| `execution.time_limit_hours_per_training` | Removed | Omit the historical resource allocation from the template. The queue retains per-job limits. |
| `execution.tmux_prefix` | Changed | Parameterize the session prefix via `P24_TMUX_PREFIX`. |
| `experiments.D0.acquisition_approval` | Added | Record the acquisition and approval gate under a neutral field name. |
| `experiments.D0.deadline` | Removed | Omit the original acquisition deadline. |
| `experiments.D0.dispatch` | Removed | Replace the old field with `experiments.D0.acquisition_approval`. Acquisition reporting and approval remain prerequisites. |
| `experiments.D0.max_hours` | Removed | Omit the original acquisition time allocation. |
| `experiments.D0.sources` | Changed | Retain the restriction to official links from the original dataset papers or appendices. Omit their internal bibliography lookup. |
| `id` | Changed | Use the neutral public registration ID. |
| `phase1_prereg` | Changed | Reference the renamed phase 1 template. |

## prereg_p24_phase3.yaml

| Field | Change | Justification |
| --- | --- | --- |
| `R2.gpu` | Removed | Omit the private host, physical device placement, lock, session naming and historical wall-clock allocation. Keep the existing replay and measurement protocol. |
| `R5.routers` | Changed | Limit the audit to configurations actually run, omitting the internal manuscript reference. |
| `deadline` | Removed | Omit the original wall-clock deadline for the execution plan. |
| `id` | Changed | Use the neutral public registration ID. |
| `resources.export_bytes_max` | Removed | Omit the archive packaging limit from the template. Export code retains its limit. |
| `resources.exported_json_file_bytes_max` | Removed | Omit the per-file packaging limit from the template. Export code retains its limit. |
| `resources.new_remote_bytes_max` | Removed | Omit the original site storage allocation. |

## prereg_p24_phase4.yaml

| Field | Change | Justification |
| --- | --- | --- |
| `D4.budget` | Changed | Omit the original GPU-hour allocation. Preserve both stop conditions and record the reason. |
| `D4.device` | Changed | Retain the configured host, exclusive use and physical GPU2 requirement. Omit internal lock path and session bookkeeping. |
| `deadline` | Removed | Omit the original wall-clock deadline for the execution plan. |
| `id` | Changed | Use the neutral public registration ID. |
| `resources.export_bytes_max` | Removed | Omit the archive packaging allocation from the template. |
| `resources.gpu_optional` | Removed | Remove the duplicated machine-specific allocation and lock details. The D4 fields retain an optional exclusive measurement and its skip conditions. |
| `resources.max_new_remote_bytes` | Removed | Omit the original site storage allocation. |
| `version` | Changed | Use a neutral public version name. |

## prereg_p24_phase5.yaml

| Field | Change | Justification |
| --- | --- | --- |
| `A6.no_go` | Changed | Omit the internal route label. Keep the immediate stop of accuracy-oriented GPU work and the prohibition on starting remaining A7 runs. |
| `A8.gpu_hours_max` | Removed | Omit the original optional timing allocation. Keep the optional measurement and skip rule. |
| `authority` | Removed | Omit reference to the internal planning document and decision workflow. |
| `data.source_repo` | Changed | Parameterize the code root via `P24_REPO_ROOT`. |
| `deadlines.audit` | Removed | Omit the internal audit delivery time. |
| `deadlines.experiments_hard_stop` | Removed | Omit the overall campaign cutoff. The A7 launch cutoff and optional experiment window are retained. |
| `deadlines.tables_evidence` | Removed | Omit the internal tables and evidence delivery time. |
| `id` | Changed | Use the neutral public registration ID. |
| `integrity.preservation` | Removed | Omit repository and worktree operational directives. Append-only ledger and cache rules remain. |
| `integrity.reporting` | Changed | Keep decisions and their evidence at each stop-loss checkpoint. Omit internal packaging instructions, filename and size cap. |
| `resources.A8_guard` | Changed | Reword the same sequence: release the training lease, inspect memory and processes, then obtain a fresh exclusive timing lease. |
| `resources.concurrent_jobs` | Removed | Omit site-specific job capacity and lock bookkeeping. The scientific A7 pilot concurrency condition is retained. |
| `resources.export_bytes_max` | Removed | Omit the archive packaging allocation. |
| `resources.host` | Changed | Parameterize the host via `P24_FORMAL_HOST`. |
| `resources.initial_guard` | Changed | Keep the memory limit and exclusion of unrelated GPU work. Omit internal process and session instructions. |
| `resources.lock` | Removed | Omit the absolute private lock path. |
| `resources.lock_content` | Removed | Omit internal lock-file bookkeeping. |
| `resources.lock_host` | Removed | Omit the private lock server. |
| `resources.tmux_prefix` | Removed | Omit the historic session prefix from this template. |
| `route` | Removed | Omit the internal decision label. Scientific stop and continue rules remain in their experiment fields. |
| `runtime_amendment_20261114_1252.cache` | Changed | Match the same exclusion entries relative to the configured feature root. Keep the replacement namespace, runtime and size rules. |
| `runtime_amendment_20261114_1252.formal_runs_root` | Changed | Parameterize the formal ledger root via `P24_RUN_ROOT`. |
| `runtime_amendment_20261114_1252.old_runs_preserved` | Changed | Parameterize the old ledger root via `P24_RUN_ROOT`. |
| `runtime_amendment_20261114_1252.python` | Changed | Parameterize the code root of the pinned executable via `P24_REPO_ROOT`. |
| `runtime_amendment_20261114_1252.unchanged_rules` | Changed | Clarify that the experiment-launch deadlines are retained. Internal delivery dates and the overall campaign cutoff are omitted. |

## prereg_p24_phase6_e1.yaml

| Field | Change | Justification |
| --- | --- | --- |
| `id` | Changed | Use the neutral public registration ID. |
| `phase5_root` | Changed | Parameterize the pinned phase 5 input ledger via `P24_RUN_ROOT`. |
| `runtime.cpu_hours_max` | Removed | Omit the original CPU allocation from the public template. |
| `runtime.python` | Changed | Parameterize the code root of the pinned executable via `P24_REPO_ROOT`. |
