# Supervisor policy

Unit: discos-minji-eqbench-large.service (systemd --user).
Grid is loaded and validated from ../plan.json, never from old default capacities.
Fresh process, DRAM and DAOS namespace for every case or interrupted-case retry.
Only explicitly scoped case KV is deleted; all logs and partial attempts remain.
Retain complete successes; retry only operational failures, never unfavorable timing.
Automatic retries: at most 3 per condition; no-progress timeout: 900 seconds.
Unknown validation/code/CUDA OOM/storage errors require diagnosis, not workload changes.
Main process and child processes share the user service cgroup for shutdown cleanup.
