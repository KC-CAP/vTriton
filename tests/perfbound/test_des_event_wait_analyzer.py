import json

from perfbound.analyze.des_event_wait_analyzer import analyze_des_event_wait


def _write_des(tmp_path, data):
    path = tmp_path / "des.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_event_wait_is_attribution_not_extra_elapsed_time(tmp_path):
    des = {
        "clock_ghz": 1.85,
        "critical_path_summary": {
            "cycles": 100,
            "issue_cycles": 70,
            "event_wait_cycles": 30,
            "ops": [0, 1],
        },
        "calibration_summary": {
            "sync_event_wait_cycles": 40,
        },
        "operations": [
            {
                "id": 0,
                "name": "sync_block_set",
                "pipe": "PIPE_MTE2_V",
                "core_type": "VECTOR",
                "issue_duration": 10,
                "event_wait_cycles": 0,
                "end_cycle": 10,
            },
            {
                "id": 1,
                "name": "sync_block_wait",
                "pipe": "PIPE_ALL",
                "core_type": "CUBE",
                "issue_duration": 5,
                "event_wait_cycles": 30,
                "end_cycle": 100,
            },
            {
                "id": 2,
                "name": "wait_flag",
                "pipe": "PIPE_V",
                "core_type": "VECTOR",
                "issue_duration": 5,
                "event_wait_cycles": 10,
                "end_cycle": 60,
            },
        ],
    }
    result = analyze_des_event_wait(
        _write_des(tmp_path, des),
        mix_block_num=205,
        profiling_e2e_ms=104.292,
    )

    assert result.critical_path_elapsed_cycles == 100
    assert result.critical_path_event_wait_cycles == 30
    assert result.block_elapsed_us == 100 / 1850.0
    assert result.e2e_elapsed_ms == (100 / 1850.0) * 205 / 1000.0
    assert result.e2e_with_event_wait_added_ms == (130 / 1850.0) * 205 / 1000.0
    assert result.event_wait_already_in_elapsed is True
    assert result.coverage_elapsed < result.coverage_if_double_counted

    full_top = result.full_wait_top[0]
    assert full_top.key == "sync_block_wait|PIPE_ALL|CUBE"
    assert full_top.wait_cycles == 30

    cp_top = result.critical_path_wait_top[0]
    assert cp_top.key == "sync_block_wait|PIPE_ALL|CUBE"
    assert cp_top.wait_cycles == 30


def test_missing_critical_path_summary_falls_back_to_max_end_cycle(tmp_path):
    des = {
        "clock_ghz": 2.0,
        "operations": [
            {"id": 10, "name": "a", "pipe": "PIPE_S", "end_cycle": 25},
            {"id": 20, "name": "b", "pipe": "PIPE_V", "end_cycle": 50},
        ],
    }
    result = analyze_des_event_wait(_write_des(tmp_path, des), mix_block_num=4)

    assert result.critical_path_elapsed_cycles == 50
    assert result.critical_path_ops == []
    assert result.block_elapsed_us == 50 / 2000.0
    assert result.e2e_elapsed_ms == (50 / 2000.0) * 4 / 1000.0
