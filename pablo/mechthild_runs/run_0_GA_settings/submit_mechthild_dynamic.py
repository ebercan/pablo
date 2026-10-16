"""Dynamic SLURM submitter for run_0 GA settings workers.

Submits CPU and GPU worker arrays that drain the file-based job queue in
``queue_state/``, sized to the free slots on each partition.

Usage:
    python -m pablo.mechthild_runs.run_0_GA_settings.submit_mechthild_dynamic [COMMAND]

COMMAND is one of ``submit`` (default, one-shot), ``status``, ``autoscale``,
``diagnostics``, ``submit_cpu_only``, ``status_cpu_only`` or
``autoscale_cpu_only``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from pablo.mechthild_runs.run_0_GA_settings.config import (
    ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES,
    GPU_EXCLUDE_NODES,
    LOGS_DIRNAME,
    QUEUE_DIRNAME,
    RESULTS_DIRNAME,
)


CPU_PARTITION = "medium"
GPU_PARTITION = "gpu"
# Backward-compatible alias used in older docs and tests.
PARTITION = CPU_PARTITION
JOB_NAME_PREFIX = "ga_set2"
CPU_JOB_SUFFIX = "cpu"
GPU_JOB_SUFFIX = "gpu"
CPUS_PER_TASK = 1
GPUS_PER_TASK = 1
GPU_GRES_REQUEST = f"gpu:{GPUS_PER_TASK}"
MEMORY = "16G"
TIME_LIMIT = "24:00:00"
EMAIL = "enisbaris@mpi-magdeburg.mpg.de"
MIN_WORKERS_FALLBACK = 10
MIN_GPU_WORKERS_FALLBACK = 0
DEFAULT_POLL_INTERVAL_SECONDS = 30
DEFAULT_FORCE_CPU_FALLBACK = 1 if ALLOW_CPU_FALLBACK_ON_CUDA_ISSUES else 0
CPU_ONLY_COMMAND_SUFFIX = "_cpu_only"
THROTTLE_REASON_TOKENS = (
    "ASSOCMAX",
    "QOSMAX",
    "MAXJOBS",
    "MAXSUBMIT",
    "JOBLIMIT",
    "NODELIMIT",
)


def _run_capture(command: list[str]) -> subprocess.CompletedProcess:
    """Run subprocess command and capture output without raising.

    Args:
        command: Shell command as argument list.

    Returns:
        CompletedProcess instance.
    """
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _has_command(command_name: str) -> bool:
    """Check if a command is available on PATH.

    Args:
        command_name: Command name, e.g. squeue.

    Returns:
        True when command is available.
    """
    return shutil.which(command_name) is not None


def _parse_leading_int(raw_value: str) -> int | None:
    """Parse a leading integer from a SLURM field value.

    Args:
        raw_value: Raw token value from SLURM output.

    Returns:
        Parsed integer, or None when no leading digits are present.
    """
    cleaned = raw_value.strip()
    if not cleaned:
        return None

    digits = []
    for char in cleaned:
        if char.isdigit():
            digits.append(char)
            continue
        break

    if not digits:
        return None
    return int("".join(digits))


def _parse_slurm_memory_to_mb(raw_value: str) -> int:
    """Parse a SLURM memory token and convert to MiB.

    Args:
        raw_value: Memory token, e.g. 16000M, 16G, 1T, or 32000.

    Returns:
        Memory amount in MiB. Returns zero for unparsable input.
    """
    cleaned = raw_value.strip().upper()
    if not cleaned:
        return 0

    # Remove optional suffixes sometimes present in TRES values.
    cleaned = cleaned.rstrip("C")

    digits = []
    suffix = ""
    for char in cleaned:
        if char.isdigit():
            digits.append(char)
            continue
        suffix = cleaned[len(digits) :].strip()
        break

    if not digits:
        return 0

    value = int("".join(digits))
    if not suffix:
        return value

    if suffix.startswith("K"):
        return max(value // 1024, 0)
    if suffix.startswith("M"):
        return value
    if suffix.startswith("G"):
        return value * 1024
    if suffix.startswith("T"):
        return value * 1024 * 1024
    if suffix.startswith("P"):
        return value * 1024 * 1024 * 1024

    return value


def _extract_field_value(line: str, field_name: str) -> str:
    """Extract one key=value token value from scontrol output.

    Args:
        line: One line from `scontrol show nodes -o` output.
        field_name: Name of the target field, e.g. CfgTRES.

    Returns:
        Field value string, or empty string when field is missing.
    """
    prefix = f"{field_name}="
    for token in line.split():
        if token.startswith(prefix):
            return token[len(prefix) :]
    return ""


def _extract_node_name(line: str) -> str:
    """Extract node name from one scontrol node line.

    Args:
        line: One line from `scontrol show nodes -o` output.

    Returns:
        Node name string, or empty string when missing.
    """
    return _extract_field_value(line, "NodeName")


def _extract_node_state(line: str) -> str:
    """Extract node state token from one scontrol node line.

    Args:
        line: One line from `scontrol show nodes -o` output.

    Returns:
        Node state string, or empty string when missing.
    """
    return _extract_field_value(line, "State")


def _is_eligible_cpu_node_state(state_value: str) -> bool:
    """Determine whether a node state is eligible for CPU scheduling.

    Args:
        state_value: Node state field value from scontrol output.

    Returns:
        True when state indicates node can run jobs.
    """
    state_upper = state_value.upper()
    if not state_upper:
        return True

    disallowed_fragments = (
        "DOWN",
        "DRAIN",
        "MAINT",
        "FAIL",
        "UNKNOWN",
        "POWERED_DOWN",
    )
    return not any(fragment in state_upper for fragment in disallowed_fragments)


def _parse_tres_blob(tres_blob: str) -> dict[str, int]:
    """Parse one TRES blob into integer values by key.

    Args:
        tres_blob: Comma-separated TRES string from SLURM.

    Returns:
        Mapping from TRES key to parsed integer value.
    """
    entries = [entry.strip() for entry in tres_blob.split(",") if entry.strip()]
    parsed: dict[str, int] = {}
    for entry in entries:
        key_value = entry.split("=", 1)
        if len(key_value) != 2:
            continue
        key, value = key_value
        count = _parse_leading_int(value)
        if count is None:
            continue
        parsed[key] = count
    return parsed


def _extract_total_gpu_count(tres_blob: str) -> int:
    """Extract total GPU count from one TRES blob.

    Args:
        tres_blob: Comma-separated TRES string from SLURM.

    Returns:
        Total GPU count inferred from typed or untyped GPU keys.
    """
    tres_map = _parse_tres_blob(tres_blob)
    typed_total = sum(
        count for key, count in tres_map.items() if key.startswith("gres/gpu:")
    )
    if typed_total > 0:
        return typed_total
    return int(tres_map.get("gres/gpu", 0))


def _extract_gpu_type_counts(tres_blob: str) -> dict[str, int]:
    """Extract per-type GPU counts from one TRES blob.

    Args:
        tres_blob: Comma-separated TRES string from SLURM.

    Returns:
        Mapping from GPU type labels to counts.
    """
    tres_map = _parse_tres_blob(tres_blob)
    type_counts: dict[str, int] = {}
    generic_count = int(tres_map.get("gres/gpu", 0))
    if generic_count > 0:
        type_counts["generic"] = generic_count

    for key, value in tres_map.items():
        if not key.startswith("gres/gpu:"):
            continue
        gpu_type = key.split(":", 1)[1].strip() or "unknown"
        type_counts[gpu_type] = int(value)

    return type_counts


def _subtract_type_counts(
    configured: dict[str, int],
    allocated: dict[str, int],
) -> dict[str, int]:
    """Compute non-negative free counts by GPU type.

    Args:
        configured: Configured counts by type.
        allocated: Allocated counts by type.

    Returns:
        Free counts by type.
    """
    free: dict[str, int] = {}
    all_types = set(configured.keys()) | set(allocated.keys())
    for gpu_type in sorted(all_types):
        free[gpu_type] = max(
            int(configured.get(gpu_type, 0)) - int(allocated.get(gpu_type, 0)),
            0,
        )
    return free


def _extract_tres_count(tres_blob: str, tres_name: str) -> int:
    """Extract one TRES count from a CfgTRES/AllocTRES blob.

    Args:
        tres_blob: Comma-separated TRES string from SLURM.
        tres_name: TRES key, e.g. cpu or gres/gpu.

    Returns:
        Parsed count for the requested TRES. Returns zero on parse failures.
    """
    entries = [entry.strip() for entry in tres_blob.split(",") if entry.strip()]

    # Prefer exact key matches first, e.g. "gres/gpu=4".
    for entry in entries:
        key_value = entry.split("=", 1)
        if len(key_value) != 2:
            continue
        key, value = key_value
        if key != tres_name:
            continue
        parsed = _parse_leading_int(value)
        if parsed is None:
            return 0
        return parsed

    # If the cluster reports typed GPUs only (e.g. gres/gpu:a100), sum them.
    if tres_name == "gres/gpu":
        total = 0
        for entry in entries:
            key_value = entry.split("=", 1)
            if len(key_value) != 2:
                continue
            key, value = key_value
            if not key.startswith("gres/gpu:"):
                continue
            parsed = _parse_leading_int(value)
            if parsed is None:
                continue
            total += parsed
        return total

    return 0


def _extract_mem_mb_from_tres_blob(tres_blob: str) -> int:
    """Extract memory value from one TRES blob in MiB.

    Args:
        tres_blob: Comma-separated TRES string from SLURM.

    Returns:
        Memory amount in MiB. Returns zero when unavailable.
    """
    entries = [entry.strip() for entry in tres_blob.split(",") if entry.strip()]
    for entry in entries:
        key_value = entry.split("=", 1)
        if len(key_value) != 2:
            continue

        key, value = key_value
        if key != "mem":
            continue

        return _parse_slurm_memory_to_mb(value)

    return 0


def _query_cpu_node_lines(partition: str) -> tuple[int, str, list[str], str]:
    """Query CPU node lines with a robust fallback chain.

    Args:
        partition: CPU partition name.

    Returns:
        Tuple (return_code, stderr, raw_lines, source_label).
    """
    primary = _run_capture(["scontrol", "show", "nodes", "-p", partition, "-o"])
    primary_lines = [
        line.strip() for line in primary.stdout.splitlines() if line.strip()
    ]
    if primary.returncode == 0 and primary_lines:
        return 0, primary.stderr.strip(), primary_lines, "scontrol_partition"

    fallback = _run_capture(["scontrol", "show", "nodes", "-o"])
    fallback_lines_all = [
        line.strip() for line in fallback.stdout.splitlines() if line.strip()
    ]
    fallback_lines = [
        line for line in fallback_lines_all if _line_has_partition(line, partition)
    ]
    if fallback.returncode == 0 and fallback_lines:
        return 0, fallback.stderr.strip(), fallback_lines, "scontrol_all_nodes"

    stderr_parts = []
    if primary.stderr.strip():
        stderr_parts.append(f"primary={primary.stderr.strip()}")
    if fallback.stderr.strip():
        stderr_parts.append(f"fallback={fallback.stderr.strip()}")
    combined_stderr = " | ".join(stderr_parts)
    return_code = primary.returncode if primary.returncode != 0 else fallback.returncode
    return return_code, combined_stderr, [], "none"


def _sum_free_gpus_from_scontrol(partition: str) -> int:
    """Estimate currently free GPUs from scontrol node TRES values.

    Args:
        partition: Partition name.

    Returns:
        Estimated number of free GPUs in the partition.
    """
    inventory = collect_gpu_inventory(partition)
    if int(inventory["query_return_code"]) != 0:
        return MIN_GPU_WORKERS_FALLBACK
    return int(inventory["total_free_gpus"])


def _line_has_partition(line: str, partition: str) -> bool:
    """Check whether one scontrol node line belongs to a partition.

    Args:
        line: One line from `scontrol show nodes -o` output.
        partition: Target partition.

    Returns:
        True when partition appears in the node's Partitions field.
    """
    partitions_blob = _extract_field_value(line, "Partitions")
    labels = [label.strip() for label in partitions_blob.split(",") if label.strip()]
    return partition in labels


def _query_gpu_node_lines(partition: str) -> tuple[int, str, list[str], str]:
    """Query GPU node lines with a robust fallback chain.

    Args:
        partition: GPU partition name.

    Returns:
        Tuple (return_code, stderr, raw_lines, source_label).
    """
    primary = _run_capture(["scontrol", "show", "nodes", "-p", partition, "-o"])
    primary_lines = [
        line.strip() for line in primary.stdout.splitlines() if line.strip()
    ]
    if primary.returncode == 0 and primary_lines:
        return 0, primary.stderr.strip(), primary_lines, "scontrol_partition"

    fallback = _run_capture(["scontrol", "show", "nodes", "-o"])
    fallback_lines_all = [
        line.strip() for line in fallback.stdout.splitlines() if line.strip()
    ]
    fallback_lines = [
        line for line in fallback_lines_all if _line_has_partition(line, partition)
    ]
    if fallback.returncode == 0 and fallback_lines:
        return 0, fallback.stderr.strip(), fallback_lines, "scontrol_all_nodes"

    stderr_parts = []
    if primary.stderr.strip():
        stderr_parts.append(f"primary={primary.stderr.strip()}")
    if fallback.stderr.strip():
        stderr_parts.append(f"fallback={fallback.stderr.strip()}")
    combined_stderr = " | ".join(stderr_parts)
    return_code = primary.returncode if primary.returncode != 0 else fallback.returncode
    return return_code, combined_stderr, [], "none"


def collect_gpu_inventory(partition: str) -> dict[str, object]:
    """Collect per-node and per-type GPU inventory diagnostics.

    Args:
        partition: GPU partition name.

    Returns:
        Dictionary with parsed node summaries and aggregate totals.
    """
    query_return_code, query_stderr, raw_lines, query_source = _query_gpu_node_lines(
        partition
    )

    node_entries: list[dict[str, object]] = []
    total_cfg = 0
    total_alloc = 0
    aggregate_cfg_types: dict[str, int] = {}
    aggregate_alloc_types: dict[str, int] = {}
    aggregate_free_types: dict[str, int] = {}

    for line in raw_lines:
        cfg_blob = _extract_field_value(line, "CfgTRES")
        alloc_blob = _extract_field_value(line, "AllocTRES")
        cfg_total = _extract_total_gpu_count(cfg_blob) if cfg_blob else 0
        alloc_total = _extract_total_gpu_count(alloc_blob) if alloc_blob else 0
        free_total = max(cfg_total - alloc_total, 0)

        cfg_types = _extract_gpu_type_counts(cfg_blob) if cfg_blob else {}
        alloc_types = _extract_gpu_type_counts(alloc_blob) if alloc_blob else {}
        free_types = _subtract_type_counts(cfg_types, alloc_types)

        for key, value in cfg_types.items():
            aggregate_cfg_types[key] = int(aggregate_cfg_types.get(key, 0)) + int(value)
        for key, value in alloc_types.items():
            aggregate_alloc_types[key] = int(aggregate_alloc_types.get(key, 0)) + int(
                value
            )
        for key, value in free_types.items():
            aggregate_free_types[key] = int(aggregate_free_types.get(key, 0)) + int(
                value
            )

        node_entries.append(
            {
                "node_name": _extract_node_name(line),
                "cfg_gpus": cfg_total,
                "alloc_gpus": alloc_total,
                "free_gpus": free_total,
                "cfg_gpu_types": cfg_types,
                "alloc_gpu_types": alloc_types,
                "free_gpu_types": free_types,
                "cfg_tres": cfg_blob,
                "alloc_tres": alloc_blob,
                "raw_line": line,
            }
        )

        total_cfg += cfg_total
        total_alloc += alloc_total

    return {
        "partition": partition,
        "query_return_code": int(query_return_code),
        "query_stderr": query_stderr,
        "query_source": query_source,
        "raw_lines": raw_lines,
        "nodes": node_entries,
        "total_cfg_gpus": total_cfg,
        "total_alloc_gpus": total_alloc,
        "total_free_gpus": max(total_cfg - total_alloc, 0),
        "cfg_gpu_types": aggregate_cfg_types,
        "alloc_gpu_types": aggregate_alloc_types,
        "free_gpu_types": aggregate_free_types,
    }


def print_gpu_inventory_diagnostics(partition: str) -> None:
    """Print submitter GPU diagnostics used by capacity logic.

    Args:
        partition: GPU partition name.
    """
    inventory = collect_gpu_inventory(partition)
    print(f"GPU diagnostics partition: {partition}")
    print(f"  query_return_code: {inventory['query_return_code']}")
    print(f"  query_source: {inventory['query_source']}")
    if inventory["query_stderr"]:
        print(f"  query_stderr: {inventory['query_stderr']}")

    print(
        "  totals: "
        f"cfg={inventory['total_cfg_gpus']} "
        f"alloc={inventory['total_alloc_gpus']} "
        f"free={inventory['total_free_gpus']}"
    )
    print(f"  free by type: {inventory['free_gpu_types']}")

    for node in inventory["nodes"]:
        print(
            "  node "
            f"{node['node_name']}: cfg={node['cfg_gpus']} "
            f"alloc={node['alloc_gpus']} free={node['free_gpus']} "
            f"free_types={node['free_gpu_types']}"
        )
        print(f"    raw: {node['raw_line']}")


def get_total_capacity(partition: str, cpus_per_task: int = CPUS_PER_TASK) -> int:
    """Estimate currently free CPU worker slots from CPU+memory node inventory.

    Args:
        partition: Partition name.
        cpus_per_task: CPUs required by each worker.

    Returns:
        Number of worker slots available right now.
    """
    if cpus_per_task <= 0:
        return MIN_WORKERS_FALLBACK

    requested_mem_mb = _parse_slurm_memory_to_mb(MEMORY)

    query_return_code, _query_stderr, raw_lines, _query_source = _query_cpu_node_lines(
        partition
    )

    if query_return_code == 0 and raw_lines:
        parsed_any_node = False
        total_slots = 0

        for line in raw_lines:
            node_state = _extract_node_state(line)
            if not _is_eligible_cpu_node_state(node_state):
                continue

            cfg_blob = _extract_field_value(line, "CfgTRES")
            if not cfg_blob:
                continue

            parsed_any_node = True
            alloc_blob = _extract_field_value(line, "AllocTRES")
            cfg_cpu = _extract_tres_count(cfg_blob, "cpu")
            alloc_cpu = _extract_tres_count(alloc_blob, "cpu") if alloc_blob else 0
            free_cpu = max(cfg_cpu - alloc_cpu, 0)
            cpu_slots = max(free_cpu // cpus_per_task, 0)

            if requested_mem_mb <= 0:
                total_slots += cpu_slots
                continue

            cfg_mem_mb = _extract_mem_mb_from_tres_blob(cfg_blob)
            alloc_mem_mb = (
                _extract_mem_mb_from_tres_blob(alloc_blob) if alloc_blob else 0
            )
            if cfg_mem_mb <= 0:
                total_slots += cpu_slots
                continue

            free_mem_mb = max(cfg_mem_mb - alloc_mem_mb, 0)
            mem_slots = max(free_mem_mb // requested_mem_mb, 0)
            total_slots += min(cpu_slots, mem_slots)

        if parsed_any_node:
            return max(total_slots, 0)

    cpu_result = _run_capture(["sinfo", "-p", partition, "-h", "-o", "%C"])
    if cpu_result.returncode != 0:
        return MIN_WORKERS_FALLBACK

    lines = [line.strip() for line in cpu_result.stdout.splitlines() if line.strip()]
    if not lines:
        return MIN_WORKERS_FALLBACK

    total_idle_cpus = 0
    for line in lines:
        parts = line.split("/")
        if len(parts) < 2:
            return MIN_WORKERS_FALLBACK
        idle_str = parts[1].strip()
        if not idle_str.isdigit():
            return MIN_WORKERS_FALLBACK
        total_idle_cpus += int(idle_str)

    return max(total_idle_cpus // cpus_per_task, 0)


def get_total_gpu_capacity(partition: str, gpus_per_task: int = GPUS_PER_TASK) -> int:
    """Estimate currently free GPU worker slots.

    Args:
        partition: GPU partition name.
        gpus_per_task: GPUs required by each worker.

    Returns:
        Number of GPU worker slots available right now.
    """
    if gpus_per_task <= 0:
        return MIN_GPU_WORKERS_FALLBACK

    inventory = collect_gpu_inventory(partition)
    if int(inventory["query_return_code"]) != 0:
        return MIN_GPU_WORKERS_FALLBACK

    total_free_gpus = int(inventory["total_free_gpus"])
    return max(total_free_gpus // gpus_per_task, 0)


def get_active_workers(job_prefix: str, partition: str | None = None) -> int:
    """Count active worker jobs with matching job-name prefix.

    Args:
        job_prefix: Prefix for SLURM job names.
        partition: Optional partition filter.

    Returns:
        Number of active workers.
    """
    user = os.environ.get("USER", "")
    if not user:
        return 0
    if not _has_command("squeue"):
        return 0

    result = _run_capture(
        ["squeue", "-u", user, "-t", "RUNNING", "--noheader", "--format=%j|%P"]
    )
    if result.returncode != 0:
        return 0

    matches = 0
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        pieces = stripped.split("|", 1)
        job_name = pieces[0].strip()
        job_partition = pieces[1].strip() if len(pieces) == 2 else ""

        if job_prefix and not job_name.startswith(job_prefix):
            continue
        if partition is not None and partition not in job_partition:
            continue

        matches += 1

    return matches


def get_pending_workers(
    job_prefix: str,
    partition: str | None = None,
) -> tuple[int, dict[str, int]]:
    """Count pending workers and summarize pending reasons.

    Args:
        job_prefix: Prefix for SLURM job names.
        partition: Optional partition filter.

    Returns:
        Tuple (pending_count, pending_reason_counts).
    """
    user = os.environ.get("USER", "")
    if not user:
        return 0, {}
    if not _has_command("squeue"):
        return 0, {}

    result = _run_capture(
        [
            "squeue",
            "-u",
            user,
            "-t",
            "PENDING",
            "--noheader",
            "--format=%j|%P|%r",
        ]
    )
    if result.returncode != 0:
        return 0, {}

    pending_count = 0
    reason_counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        pieces = stripped.split("|", 2)
        if len(pieces) < 2:
            continue

        job_name = pieces[0].strip()
        job_partition = pieces[1].strip()
        pending_reason = pieces[2].strip() if len(pieces) == 3 else "Unknown"

        if job_prefix and not job_name.startswith(job_prefix):
            continue
        if partition is not None and partition not in job_partition:
            continue

        pending_count += 1
        reason_key = pending_reason or "Unknown"
        reason_counts[reason_key] = int(reason_counts.get(reason_key, 0)) + 1

    return pending_count, reason_counts


def _is_submission_throttle_reason(reason_text: str) -> bool:
    """Check whether one pending reason indicates scheduler throttling.

    Args:
        reason_text: Pending reason from squeue.

    Returns:
        True when reason likely reflects submission/job-account limits.
    """
    normalized = reason_text.upper().replace(" ", "")
    return any(token in normalized for token in THROTTLE_REASON_TOKENS)


def _has_submission_throttle(reason_counts: dict[str, int]) -> bool:
    """Check whether pending reasons contain throttling signals.

    Args:
        reason_counts: Mapping from reason string to count.

    Returns:
        True when at least one reason indicates scheduler throttling.
    """
    return any(_is_submission_throttle_reason(reason) for reason in reason_counts)


def get_pending_jobs(queue_dir: str | Path) -> int:
    """Count pending lock files.

    Args:
        queue_dir: queue_state path.

    Returns:
        Pending lock count.
    """
    pending_dir = Path(queue_dir) / "pending"
    if not pending_dir.exists():
        return 0
    return len(list(pending_dir.glob("*.lock")))


def _resolve_command_mode(command: str, cpu_only_flag: bool) -> tuple[str, bool]:
    """Resolve command aliases and cpu_only mode flag.

    Args:
        command: Raw command token from argparse.
        cpu_only_flag: Value from --cpu_only flag.

    Returns:
        Tuple (base_command, cpu_only_enabled).
    """
    is_cpu_only_alias = command.endswith(CPU_ONLY_COMMAND_SUFFIX)
    if is_cpu_only_alias:
        base_command = command[: -len(CPU_ONLY_COMMAND_SUFFIX)]
    else:
        base_command = command
    return base_command, bool(cpu_only_flag or is_cpu_only_alias)


def get_capacity_snapshot(
    queue_dir: str | Path,
    cpu_partition: str,
    gpu_partition: str,
    job_prefix: str = JOB_NAME_PREFIX,
    cpu_only: bool = False,
) -> dict[str, int]:
    """Collect queue and dual-pool free-slot metrics for one instant.

    Args:
        queue_dir: Path to queue_state directory.
        cpu_partition: CPU partition name.
        gpu_partition: GPU partition name.
        job_prefix: Prefix used to count this run's worker jobs.
        cpu_only: If True, skip all GPU capacity and active-worker checks.

    Returns:
        Dictionary with CPU/GPU free slots, active worker telemetry,
        pending jobs, and planned submissions.
    """
    pending_jobs = get_pending_jobs(queue_dir)
    cpu_free_slots_now = get_total_capacity(cpu_partition)
    gpu_free_slots_now = 0 if cpu_only else get_total_gpu_capacity(gpu_partition)

    cpu_active_workers = get_active_workers(
        job_prefix=f"{job_prefix}_{CPU_JOB_SUFFIX}", partition=cpu_partition
    )
    cpu_pending_workers, cpu_pending_reasons = get_pending_workers(
        job_prefix=f"{job_prefix}_{CPU_JOB_SUFFIX}", partition=cpu_partition
    )
    gpu_active_workers = 0
    gpu_pending_workers = 0
    gpu_pending_reasons: dict[str, int] = {}
    if not cpu_only:
        gpu_active_workers = get_active_workers(
            job_prefix=f"{job_prefix}_{GPU_JOB_SUFFIX}", partition=gpu_partition
        )
        gpu_pending_workers, gpu_pending_reasons = get_pending_workers(
            job_prefix=f"{job_prefix}_{GPU_JOB_SUFFIX}", partition=gpu_partition
        )

    cpu_available_slots = max(cpu_free_slots_now, 0)
    gpu_available_slots = 0 if cpu_only else max(gpu_free_slots_now, 0)

    cpu_submission_throttled = _has_submission_throttle(cpu_pending_reasons)
    gpu_submission_throttled = (
        False if cpu_only else _has_submission_throttle(gpu_pending_reasons)
    )

    cpu_effective_slots = max(cpu_available_slots - cpu_pending_workers, 0)
    gpu_effective_slots = (
        0 if cpu_only else max(gpu_available_slots - gpu_pending_workers, 0)
    )

    if cpu_submission_throttled:
        cpu_effective_slots = 0
    if gpu_submission_throttled:
        gpu_effective_slots = 0

    workers_to_submit_gpu = 0 if cpu_only else min(pending_jobs, gpu_effective_slots)
    remaining_after_gpu = max(pending_jobs - workers_to_submit_gpu, 0)
    workers_to_submit_cpu = min(remaining_after_gpu, cpu_effective_slots)

    estimated_capacity = cpu_free_slots_now + gpu_free_slots_now
    active_workers = cpu_active_workers + gpu_active_workers
    pending_workers = cpu_pending_workers + gpu_pending_workers
    available_slots_raw = cpu_available_slots + gpu_available_slots
    available_slots = cpu_effective_slots + gpu_effective_slots
    workers_to_submit = workers_to_submit_gpu + workers_to_submit_cpu

    return {
        "pending_jobs": pending_jobs,
        "estimated_capacity": estimated_capacity,
        "active_workers": active_workers,
        "pending_workers": pending_workers,
        "available_slots_raw": available_slots_raw,
        "available_slots": available_slots,
        "workers_to_submit": workers_to_submit,
        "cpu_estimated_capacity": cpu_free_slots_now,
        "gpu_estimated_capacity": gpu_free_slots_now,
        "cpu_active_workers": cpu_active_workers,
        "gpu_active_workers": gpu_active_workers,
        "cpu_pending_workers": cpu_pending_workers,
        "gpu_pending_workers": gpu_pending_workers,
        "cpu_pending_reasons": cpu_pending_reasons,
        "gpu_pending_reasons": gpu_pending_reasons,
        "cpu_available_slots": cpu_available_slots,
        "gpu_available_slots": gpu_available_slots,
        "cpu_effective_slots": cpu_effective_slots,
        "gpu_effective_slots": gpu_effective_slots,
        "cpu_submission_throttled": cpu_submission_throttled,
        "gpu_submission_throttled": gpu_submission_throttled,
        "workers_to_submit_cpu": workers_to_submit_cpu,
        "workers_to_submit_gpu": workers_to_submit_gpu,
    }


def print_capacity_status(
    snapshot: dict[str, int],
    cpu_partition: str,
    gpu_partition: str,
    cpu_only: bool = False,
) -> None:
    """Print compact dual-pool capacity status for operators.

    Args:
        snapshot: Capacity snapshot dictionary produced by get_capacity_snapshot.
        cpu_partition: CPU partition name.
        gpu_partition: GPU partition name.
        cpu_only: If True, report only CPU mode and skip GPU lines.
    """
    if cpu_only:
        print("Mode: cpu_only (GPU checks skipped)")

    cpu_pending_workers = int(snapshot.get("cpu_pending_workers", 0))
    gpu_pending_workers = int(snapshot.get("gpu_pending_workers", 0))
    cpu_effective_slots = int(
        snapshot.get("cpu_effective_slots", snapshot.get("cpu_available_slots", 0))
    )
    gpu_effective_slots = int(
        snapshot.get("gpu_effective_slots", snapshot.get("gpu_available_slots", 0))
    )
    cpu_submission_throttled = bool(snapshot.get("cpu_submission_throttled", False))
    gpu_submission_throttled = bool(snapshot.get("gpu_submission_throttled", False))
    cpu_pending_reasons = dict(snapshot.get("cpu_pending_reasons", {}))
    gpu_pending_reasons = dict(snapshot.get("gpu_pending_reasons", {}))
    pending_workers_total = int(
        snapshot.get("pending_workers", cpu_pending_workers + gpu_pending_workers)
    )

    print(f"CPU partition: {cpu_partition}")
    print(
        "  Free slots snapshot now "
        f"(sinfo idle / cpus_per_task): {snapshot['cpu_estimated_capacity']}"
    )
    print(f"  Workers in use: {snapshot['cpu_active_workers']}")
    print(f"  Workers pending launch: {cpu_pending_workers}")
    print(f"  Workers available now: {snapshot['cpu_available_slots']}")
    print(f"  Effective submit slots now: {cpu_effective_slots}")
    if cpu_submission_throttled:
        print(f"  Submission throttled by pending reasons: {cpu_pending_reasons}")
    if not cpu_only:
        print(f"GPU partition: {gpu_partition}")
        print(
            "  Free slots snapshot now "
            f"(parsed free gpus / gpus_per_task): {snapshot['gpu_estimated_capacity']}"
        )
        print(f"  Workers in use: {snapshot['gpu_active_workers']}")
        print(f"  Workers pending launch: {gpu_pending_workers}")
        print(f"  Workers available now: {snapshot['gpu_available_slots']}")
        print(f"  Effective submit slots now: {gpu_effective_slots}")
        if gpu_submission_throttled:
            print(
                "  Submission throttled by pending reasons: " f"{gpu_pending_reasons}"
            )
    print(f"Workers in use total: {snapshot['active_workers']}")
    print(f"Workers pending launch total: {pending_workers_total}")
    print(f"Workers available now total: {snapshot['available_slots']}")
    print(f"Pending jobs: {snapshot['pending_jobs']}")


def run_submit_once(
    queue_dir: str | Path,
    cpu_partition: str,
    gpu_partition: str,
    force: bool,
    cpu_only: bool = False,
) -> None:
    """Run one submission decision cycle.

    Args:
        queue_dir: Path to queue_state directory.
        cpu_partition: CPU partition name.
        gpu_partition: GPU partition name.
        force: If True, skip confirmation prompt.
        cpu_only: If True, submit CPU workers only.
    """
    snapshot = get_capacity_snapshot(
        queue_dir=queue_dir,
        cpu_partition=cpu_partition,
        gpu_partition=gpu_partition,
        cpu_only=cpu_only,
    )

    if snapshot["pending_jobs"] == 0:
        print("No pending jobs found.")
        print("Initialize queue first:")
        print(
            "  python -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler init"
        )
        return

    print_capacity_status(
        snapshot=snapshot,
        cpu_partition=cpu_partition,
        gpu_partition=gpu_partition,
        cpu_only=cpu_only,
    )

    if snapshot["workers_to_submit"] <= 0:
        print("No available worker slots.")
        return

    if cpu_only:
        print(f"Workers to submit (CPU only): cpu={snapshot['workers_to_submit_cpu']}")
    else:
        print(
            "Workers to submit (GPU first): "
            f"gpu={snapshot['workers_to_submit_gpu']} "
            f"cpu={snapshot['workers_to_submit_cpu']}"
        )

    if not force:
        proceed = input("Proceed? [y/N]: ")
        if proceed.lower() != "y":
            print("Cancelled.")
            return

    if (not cpu_only) and snapshot["workers_to_submit_gpu"] > 0:
        submit_array_job(
            num_workers=snapshot["workers_to_submit_gpu"],
            queue_dir=queue_dir,
            partition=gpu_partition,
            pool=GPU_JOB_SUFFIX,
        )

    if snapshot["workers_to_submit_cpu"] > 0:
        submit_array_job(
            num_workers=snapshot["workers_to_submit_cpu"],
            queue_dir=queue_dir,
            partition=cpu_partition,
            pool=CPU_JOB_SUFFIX,
        )


def run_autoscale(
    queue_dir: str | Path,
    cpu_partition: str,
    gpu_partition: str,
    poll_interval_seconds: int,
    cpu_only: bool = False,
) -> None:
    """Top up workers every poll interval until no pending jobs remain.

    Args:
        queue_dir: Path to queue_state directory.
        cpu_partition: CPU partition name.
        gpu_partition: GPU partition name.
        poll_interval_seconds: Seconds between capacity checks.
        cpu_only: If True, submit CPU workers only.

    Raises:
        ValueError: If ``poll_interval_seconds`` is not positive.
    """
    if poll_interval_seconds <= 0:
        raise ValueError("poll_interval_seconds must be greater than zero.")

    mode_label = "cpu_only" if cpu_only else "gpu_first"
    print(
        "Autoscale started. "
        f"mode={mode_label}. Polling every {poll_interval_seconds}s; press Ctrl+C to stop."
    )

    while True:
        snapshot = get_capacity_snapshot(
            queue_dir=queue_dir,
            cpu_partition=cpu_partition,
            gpu_partition=gpu_partition,
            cpu_only=cpu_only,
        )

        print_capacity_status(
            snapshot=snapshot,
            cpu_partition=cpu_partition,
            gpu_partition=gpu_partition,
            cpu_only=cpu_only,
        )

        if snapshot["pending_jobs"] == 0:
            print("No pending jobs left. Autoscale exiting.")
            return

        if (not cpu_only) and snapshot["workers_to_submit_gpu"] > 0:
            print(f"Submitting {snapshot['workers_to_submit_gpu']} GPU workers.")
            submit_array_job(
                num_workers=snapshot["workers_to_submit_gpu"],
                queue_dir=queue_dir,
                partition=gpu_partition,
                pool=GPU_JOB_SUFFIX,
            )

        if snapshot["workers_to_submit_cpu"] > 0:
            print(f"Submitting {snapshot['workers_to_submit_cpu']} CPU workers.")
            submit_array_job(
                num_workers=snapshot["workers_to_submit_cpu"],
                queue_dir=queue_dir,
                partition=cpu_partition,
                pool=CPU_JOB_SUFFIX,
            )

        if snapshot["workers_to_submit"] <= 0:
            cpu_effective_slots = int(
                snapshot.get(
                    "cpu_effective_slots", snapshot.get("cpu_available_slots", 0)
                )
            )
            gpu_effective_slots = int(
                snapshot.get(
                    "gpu_effective_slots", snapshot.get("gpu_available_slots", 0)
                )
            )
            print(
                "No worker slots available right now "
                f"(cpu_free={snapshot['cpu_available_slots']}, "
                f"gpu_free={snapshot['gpu_available_slots']}, "
                f"cpu_effective={cpu_effective_slots}, "
                f"gpu_effective={gpu_effective_slots}, "
                f"pending={snapshot['pending_jobs']})."
            )

        print(f"Sleeping for {poll_interval_seconds}s before next check.\n")
        time.sleep(poll_interval_seconds)


def submit_array_job(
    num_workers: int,
    queue_dir: str | Path,
    partition: str,
    pool: str,
) -> None:
    """Submit SLURM array workers.

    Args:
        num_workers: Number of workers to launch.
        queue_dir: queue_state path.
        partition: SLURM partition.
        pool: Execution pool, either cpu or gpu.

    Raises:
        ValueError: If ``pool`` is neither ``"cpu"`` nor ``"gpu"``.
    """
    if num_workers <= 0:
        print("No workers to submit.")
        return

    if pool not in {CPU_JOB_SUFFIX, GPU_JOB_SUFFIX}:
        raise ValueError("pool must be either 'cpu' or 'gpu'.")

    base_dir = Path(__file__).parent
    repo_root = base_dir.parents[2]
    results_dir = base_dir / RESULTS_DIRNAME
    logs_dir = base_dir / LOGS_DIRNAME
    script_path = base_dir / f".submit_dynamic_{pool}.sh"

    gres_line = f"#SBATCH --gres={GPU_GRES_REQUEST}\n" if pool == GPU_JOB_SUFFIX else ""
    exclude_line = (
        f"#SBATCH --exclude={GPU_EXCLUDE_NODES}\n" if pool == GPU_JOB_SUFFIX else ""
    )
    worker_device = "cuda" if pool == GPU_JOB_SUFFIX else "cpu"
    job_name = f"{JOB_NAME_PREFIX}_{pool}"

    script = f"""#!/bin/bash
#SBATCH -J {job_name}
#SBATCH --array=1-{num_workers}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem={MEMORY}
#SBATCH --time={TIME_LIMIT}
#SBATCH --partition={partition}
{gres_line}{exclude_line}#SBATCH -o {logs_dir}/worker_%A_%a.out
#SBATCH -e {logs_dir}/worker_%A_%a.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user={EMAIL}

set -euo pipefail

REPO_ROOT="{repo_root}"

mkdir -p "{logs_dir}"
mkdir -p "{results_dir}"
mkdir -p "{queue_dir}"

cd "$REPO_ROOT"

if [ -x "$REPO_ROOT/venv/bin/python" ]; then
    RUN2_WORKER_PYTHON="$REPO_ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    RUN2_WORKER_PYTHON="$(command -v python3)"
else
    echo "No python interpreter available for run_0 worker."
    exit 1
fi

if [ -f "$REPO_ROOT/venv/bin/activate" ]; then
    source "$REPO_ROOT/venv/bin/activate"
fi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_NUM_THREADS=$SLURM_CPUS_PER_TASK
export RUN2_WORKER_DEVICE={worker_device}
export RUN2_WORKER_ALLOW_CPU_FALLBACK={DEFAULT_FORCE_CPU_FALLBACK}
export RUN2_WORKER_PYTHON

sleep $((RANDOM % 10))

"$RUN2_WORKER_PYTHON" -m pablo.mechthild_runs.run_0_GA_settings.mechthild_scheduler worker \\
    --queue_dir "{queue_dir}"
"""

    script_path.write_text(script, encoding="utf-8")

    submit_result = _run_capture(["sbatch", str(script_path)])
    if submit_result.returncode == 0:
        print(submit_result.stdout.strip())
        print(f"Submitted {num_workers} workers")
    else:
        print(submit_result.stderr.strip())
        print("sbatch submission failed")

    if script_path.exists():
        script_path.unlink()


def run_diagnostics(
    queue_dir: str | Path,
    cpu_partition: str,
    gpu_partition: str,
) -> None:
    """Print diagnostics used for dynamic scheduling decisions.

    Args:
        queue_dir: Path to queue_state directory.
        cpu_partition: CPU partition name.
        gpu_partition: GPU partition name.
    """
    snapshot = get_capacity_snapshot(
        queue_dir=queue_dir,
        cpu_partition=cpu_partition,
        gpu_partition=gpu_partition,
    )
    print_capacity_status(
        snapshot=snapshot,
        cpu_partition=cpu_partition,
        gpu_partition=gpu_partition,
    )

    print("\nSubmitter GPU inventory diagnostics")
    print_gpu_inventory_diagnostics(gpu_partition)

    sinfo_gpu = _run_capture(["sinfo", "-p", gpu_partition, "-h", "-o", "%N|%G|%t|%C"])
    print("\nRaw sinfo gpu view")
    if sinfo_gpu.stdout.strip():
        print(sinfo_gpu.stdout.strip())
    if sinfo_gpu.stderr.strip():
        print(sinfo_gpu.stderr.strip())

    user = os.environ.get("USER", "")
    if user:
        squeue_user = _run_capture(
            [
                "squeue",
                "-u",
                user,
                "--noheader",
                "--format=%.18i|%.30j|%.20P|%.10T|%.8M|%.20R",
            ]
        )
        print("\nRaw squeue user view")
        if squeue_user.stdout.strip():
            print(squeue_user.stdout.strip())
        if squeue_user.stderr.strip():
            print(squeue_user.stderr.strip())


def main() -> None:
    """CLI entry point for dynamic worker submission."""
    default_queue_dir = str(Path(__file__).parent / QUEUE_DIRNAME)

    parser = argparse.ArgumentParser(
        description="Dynamically submit run_0 GA settings SLURM workers"
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="submit",
        choices=[
            "submit",
            "status",
            "autoscale",
            "diagnostics",
            "submit_cpu_only",
            "status_cpu_only",
            "autoscale_cpu_only",
        ],
        help=(
            "Command mode: submit (one-shot), status (print counts), "
            "autoscale (poll every interval and top up), "
            "diagnostics (show raw parser inputs), "
            "or explicit CPU-only aliases (*_cpu_only)."
        ),
    )
    parser.add_argument(
        "--queue_dir",
        default=default_queue_dir,
        help="Path to queue_state directory",
    )
    parser.add_argument(
        "--partition",
        default=CPU_PARTITION,
        help=f"CPU partition (default: {CPU_PARTITION})",
    )
    parser.add_argument(
        "--gpu_partition",
        default=GPU_PARTITION,
        help=f"GPU partition (default: {GPU_PARTITION})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt for one-shot submit",
    )
    parser.add_argument(
        "--cpu_only",
        action="store_true",
        help="Use CPU-only mode with existing submit/status/autoscale commands",
    )
    parser.add_argument(
        "--poll_interval_seconds",
        type=int,
        default=DEFAULT_POLL_INTERVAL_SECONDS,
        help=(
            "Autoscale polling interval in seconds "
            f"(default: {DEFAULT_POLL_INTERVAL_SECONDS})"
        ),
    )

    args = parser.parse_args()

    command, cpu_only = _resolve_command_mode(args.command, args.cpu_only)

    if command == "status":
        snapshot = get_capacity_snapshot(
            queue_dir=args.queue_dir,
            cpu_partition=args.partition,
            gpu_partition=args.gpu_partition,
            cpu_only=cpu_only,
        )
        print_capacity_status(
            snapshot=snapshot,
            cpu_partition=args.partition,
            gpu_partition=args.gpu_partition,
            cpu_only=cpu_only,
        )
        return

    if command == "autoscale":
        run_autoscale(
            queue_dir=args.queue_dir,
            cpu_partition=args.partition,
            gpu_partition=args.gpu_partition,
            poll_interval_seconds=args.poll_interval_seconds,
            cpu_only=cpu_only,
        )
        return

    if command == "diagnostics":
        run_diagnostics(
            queue_dir=args.queue_dir,
            cpu_partition=args.partition,
            gpu_partition=args.gpu_partition,
        )
        return

    run_submit_once(
        queue_dir=args.queue_dir,
        cpu_partition=args.partition,
        gpu_partition=args.gpu_partition,
        force=args.force,
        cpu_only=cpu_only,
    )


if __name__ == "__main__":
    main()
