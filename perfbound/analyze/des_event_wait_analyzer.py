"""Analyze DES event-wait attribution without double-counting elapsed time.

The C++ DES scheduler already emits an elapsed critical path.  Event wait
cycles explain where that elapsed time came from; they are attribution, not an
extra term to add to the elapsed critical path by default.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class WaitGroup:
    key: str
    ops: int
    wait_cycles: int
    max_wait_cycles: int


@dataclass(frozen=True)
class DesEventWaitAnalysis:
    des_path: Path
    clock_ghz: float
    mix_block_num: int
    profiling_e2e_ms: float | None
    critical_path_elapsed_cycles: int
    critical_path_issue_cycles: int
    critical_path_event_wait_cycles: int
    full_event_wait_cycles: int
    critical_path_ops: list[int]
    block_elapsed_us: float
    block_if_double_counted_us: float
    e2e_elapsed_ms: float
    e2e_with_event_wait_added_ms: float
    coverage_elapsed: float | None
    coverage_if_double_counted: float | None
    event_wait_already_in_elapsed: bool
    full_wait_top: list[WaitGroup] = field(default_factory=list)
    critical_path_wait_top: list[WaitGroup] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _operation_id(op: dict[str, Any], fallback: int) -> int:
    return _to_int(op.get("id"), fallback)


def _ops_from_des(data: dict[str, Any]) -> list[dict[str, Any]]:
    ops = data.get("operations")
    if isinstance(ops, list):
        return [op for op in ops if isinstance(op, dict)]
    nodes = data.get("nodes")
    if isinstance(nodes, list):
        return [op for op in nodes if isinstance(op, dict)]
    return []


def _critical_path_ops(data: dict[str, Any], ops: list[dict[str, Any]]) -> list[int]:
    summary = data.get("critical_path_summary")
    if isinstance(summary, dict) and isinstance(summary.get("ops"), list):
        return [_to_int(op_id) for op_id in summary["ops"]]
    return []


def _max_end_cycle(ops: Iterable[dict[str, Any]]) -> int:
    return max((_to_int(op.get("end_cycle")) for op in ops), default=0)


def _sum_ops_field(
    ops_by_id: dict[int, dict[str, Any]], op_ids: Iterable[int], field_name: str
) -> int:
    total = 0
    for op_id in op_ids:
        op = ops_by_id.get(op_id)
        if op is not None:
            total += _to_int(op.get(field_name))
    return total


def _group_waits(ops: Iterable[dict[str, Any]], limit: int) -> list[WaitGroup]:
    grouped: dict[str, list[int]] = {}
    for op in ops:
        wait = _to_int(op.get("event_wait_cycles"))
        if wait <= 0:
            continue
        key = "|".join(
            [
                str(op.get("name") or "?"),
                str(op.get("pipe") or "PIPE_UNKNOWN"),
                str(op.get("core_type") or ""),
            ]
        )
        grouped.setdefault(key, []).append(wait)
    rows = [
        WaitGroup(
            key=key,
            ops=len(values),
            wait_cycles=sum(values),
            max_wait_cycles=max(values),
        )
        for key, values in grouped.items()
    ]
    rows.sort(key=lambda row: (-row.wait_cycles, row.key))
    return rows[:limit]


def analyze_des_event_wait(
    des_path: str | Path,
    *,
    mix_block_num: int = 1,
    profiling_e2e_ms: float | None = None,
    top_n: int = 10,
) -> DesEventWaitAnalysis:
    path = Path(des_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    ops = _ops_from_des(data)
    ops_by_id = {_operation_id(op, idx): op for idx, op in enumerate(ops)}
    warnings: list[str] = []
    if not ops:
        warnings.append("DES graph has no operations/nodes array")

    clock_ghz = _to_float(data.get("clock_ghz"), 1.85)
    cycles_per_us = clock_ghz * 1000.0
    if cycles_per_us <= 0:
        warnings.append("clock_ghz <= 0; falling back to 1.85 GHz")
        clock_ghz = 1.85
        cycles_per_us = 1850.0

    critical = data.get("critical_path_summary")
    critical_ops = _critical_path_ops(data, ops)
    if isinstance(critical, dict):
        elapsed_cycles = _to_int(critical.get("cycles"), _max_end_cycle(ops))
        issue_cycles = _to_int(
            critical.get("issue_cycles"),
            _sum_ops_field(ops_by_id, critical_ops, "issue_duration"),
        )
        cp_event_wait = _to_int(
            critical.get("event_wait_cycles"),
            _sum_ops_field(ops_by_id, critical_ops, "event_wait_cycles"),
        )
    else:
        warnings.append("critical_path_summary missing; using max end_cycle fallback")
        elapsed_cycles = _max_end_cycle(ops)
        issue_cycles = 0
        cp_event_wait = 0

    calibration = data.get("calibration_summary")
    if isinstance(calibration, dict):
        full_event_wait = _to_int(calibration.get("sync_event_wait_cycles"))
    else:
        full_event_wait = sum(_to_int(op.get("event_wait_cycles")) for op in ops)

    mix_blocks = max(1, int(mix_block_num or 1))
    block_elapsed_us = elapsed_cycles / cycles_per_us
    block_double_us = (elapsed_cycles + cp_event_wait) / cycles_per_us
    e2e_elapsed_ms = block_elapsed_us * mix_blocks / 1000.0
    e2e_double_ms = block_double_us * mix_blocks / 1000.0
    coverage = None
    coverage_double = None
    if profiling_e2e_ms and profiling_e2e_ms > 0:
        coverage = e2e_elapsed_ms / profiling_e2e_ms
        coverage_double = e2e_double_ms / profiling_e2e_ms

    cp_set = set(critical_ops)
    cp_ops = [op for op_id, op in ops_by_id.items() if op_id in cp_set]

    return DesEventWaitAnalysis(
        des_path=path,
        clock_ghz=clock_ghz,
        mix_block_num=mix_blocks,
        profiling_e2e_ms=profiling_e2e_ms,
        critical_path_elapsed_cycles=elapsed_cycles,
        critical_path_issue_cycles=issue_cycles,
        critical_path_event_wait_cycles=cp_event_wait,
        full_event_wait_cycles=full_event_wait,
        critical_path_ops=critical_ops,
        block_elapsed_us=block_elapsed_us,
        block_if_double_counted_us=block_double_us,
        e2e_elapsed_ms=e2e_elapsed_ms,
        e2e_with_event_wait_added_ms=e2e_double_ms,
        coverage_elapsed=coverage,
        coverage_if_double_counted=coverage_double,
        event_wait_already_in_elapsed=True,
        full_wait_top=_group_waits(ops, top_n),
        critical_path_wait_top=_group_waits(cp_ops, top_n),
        warnings=warnings,
    )


def _pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100.0:.1f}%"


def _wait_table(rows: list[WaitGroup]) -> str:
    if not rows:
        return "_No non-zero event wait rows._\n"
    lines = [
        "| key | ops | wait cycles | max wait |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.key} | {row.ops} | {row.wait_cycles} | {row.max_wait_cycles} |"
        )
    return "\n".join(lines) + "\n"


def render_markdown(result: DesEventWaitAnalysis, title: str) -> str:
    profiling = (
        f"{result.profiling_e2e_ms:.3f} ms"
        if result.profiling_e2e_ms is not None
        else "n/a"
    )
    warnings = "\n".join(f"- {warning}" for warning in result.warnings) or "- none"
    return f"""# {title}

## Summary

| Metric | Value |
| --- | ---: |
| DES path | `{result.des_path}` |
| clock | {result.clock_ghz:.3f} GHz |
| mix_block_num | {result.mix_block_num} |
| critical path elapsed | {result.critical_path_elapsed_cycles} cycles / {result.block_elapsed_us:.3f} us |
| critical path issue cycles | {result.critical_path_issue_cycles} |
| critical path event wait cycles | {result.critical_path_event_wait_cycles} |
| full graph event wait cycles | {result.full_event_wait_cycles} |
| E2E elapsed estimate | {result.e2e_elapsed_ms:.3f} ms |
| profiling E2E | {profiling} |
| elapsed coverage | {_pct(result.coverage_elapsed)} |

## Robust Timing Rule

`critical_path_event_wait_cycles` is attribution inside the elapsed DES critical path.
It must not be added to `critical_path_elapsed_cycles` by default. The double-count
diagnostic below is emitted only to make accidental over-counting visible.

| Diagnostic | Value |
| --- | ---: |
| block if event wait is added again | {result.block_if_double_counted_us:.3f} us |
| E2E if event wait is added again | {result.e2e_with_event_wait_added_ms:.3f} ms |
| coverage if double-counted | {_pct(result.coverage_if_double_counted)} |

## Full Graph Event Wait Top

{_wait_table(result.full_wait_top)}
## Critical Path Event Wait Top

{_wait_table(result.critical_path_wait_top)}
## Warnings

{warnings}
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--des", required=True, help="Path to DES JSON")
    parser.add_argument("--mix-block-num", type=int, default=1)
    parser.add_argument("--profiling-e2e-ms", type=float)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--title", default="DES Event Wait Analysis")
    parser.add_argument("--out-md", help="Optional markdown output path")
    args = parser.parse_args(argv)

    result = analyze_des_event_wait(
        args.des,
        mix_block_num=args.mix_block_num,
        profiling_e2e_ms=args.profiling_e2e_ms,
        top_n=args.top_n,
    )
    text = render_markdown(result, args.title)
    if args.out_md:
        Path(args.out_md).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
