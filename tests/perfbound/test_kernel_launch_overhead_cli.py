import re
import subprocess
from pathlib import Path

try:
    import pytest
except ModuleNotFoundError:
    pytest = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _find_tritonsim_opt() -> Path | None:
    candidates = [
        PROJECT_ROOT / "build" / "bin" / "tritonsim-opt",
        PROJECT_ROOT / "build_hivm_llvm22" / "bin" / "tritonsim-opt",
        PROJECT_ROOT / "build_hivm_b5cc" / "bin" / "tritonsim-opt",
        PROJECT_ROOT / "build_arm64_llvm19" / "bin" / "tritonsim-opt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _require_tritonsim_opt() -> Path:
    tritonsim_opt = _find_tritonsim_opt()
    if tritonsim_opt is None:
        if pytest is not None:
            pytest.skip("tritonsim-opt binary not found")
        raise RuntimeError("tritonsim-opt binary not found")
    return tritonsim_opt


def _run_pipeline(block_dim: int, kernel_mode: str, *extra_passes: str) -> subprocess.CompletedProcess[str]:
    tritonsim_opt = _require_tritonsim_opt()
    mlir_file = PROJECT_ROOT / "test" / "transfer_spaces_ascend.mlir"
    hw_config = PROJECT_ROOT / "configs" / "ascend_910b3_v4.json"
    assert mlir_file.exists()
    assert hw_config.exists()

    cmd = [
        str(tritonsim_opt),
        str(mlir_file),
        "-assign-op-ids",
        (
            "--analyze-pipeline="
            f"hardware-config={hw_config} "
            f"arg-bindings=block_dim={block_dim},kernel_mode={kernel_mode}"
        ),
        *extra_passes,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=30)


def test_ttir_pipeline_reports_kernel_launch_overhead_from_block_dim():
    result = _run_pipeline(4, "aiv")
    assert result.returncode == 0, result.stderr

    # ascend_910b3_v4.json has explicit launch model:
    # fixed 1.0 us + 0.04 us/block * 4 blocks = 1.16 us.
    # 1.16 us * 1.85 GHz * 1000 cycles/us = 2146 cycles.
    match = re.search(r"ascend\.kernel_launch_overhead_cycles = (\d+)", result.stdout)
    assert match, result.stdout
    assert int(match.group(1)) == 2146
    assert "ascend.predicted_total_cycles" in result.stdout
    assert "Kernel launch overhead: 2146" in result.stdout


def test_ttir_pipeline_accepts_kernel_mode_aliases():
    expected = {
        "aic": 3330,
        "cube": 3330,
        "aiv": 4218,
        "vector": 4218,
    }
    for mode, cycles in expected.items():
        result = _run_pipeline(32, mode)
        assert result.returncode == 0, result.stderr
        match = re.search(r"ascend\.kernel_launch_overhead_cycles = (\d+)", result.stdout)
        assert match, result.stdout
        assert int(match.group(1)) == cycles


def test_perf_report_uses_predicted_total_when_launch_is_present():
    result = _run_pipeline(4, "aiv", "--perf-report")
    assert result.returncode == 0, result.stderr
    assert "Kernel Body Cycles:" in result.stdout
    assert "Launch Cycles:" in result.stdout
    match = re.search(r"Total Cycles:\s+(\d+)", result.stdout)
    assert match, result.stdout
    assert int(match.group(1)) == 2148


if __name__ == "__main__":
    test_ttir_pipeline_reports_kernel_launch_overhead_from_block_dim()
    test_ttir_pipeline_accepts_kernel_mode_aliases()
    test_perf_report_uses_predicted_total_when_launch_is_present()
    print("kernel launch overhead CLI smoke test passed")
