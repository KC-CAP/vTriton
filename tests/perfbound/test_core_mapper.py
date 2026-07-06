import pytest

from perfbound.distribution.core_mapper import CoreMapper


def test_mix_aic_paired_block_strategy_matches_legacy_map():
    mapper = CoreMapper(aic_cores=20, aiv_cores=40)

    result = mapper.map(
        grid=(128, 32),
        per_block_span_aic_us=11.625,
        per_block_span_aiv_us=28.455,
        task_type="MIX_AIC",
        strategy="paired_block",
    )

    assert result.total_blocks == 4096
    assert result.blocks_aic == 4096
    assert result.blocks_aiv == 4096
    assert result.waves_aic == 205
    assert result.waves_aiv == 103
    assert result.e2e_wall_us == pytest.approx(2930.865)
    assert result.strategy == "paired_block"
    assert result.assumptions


def test_mix_aic_mix_block_total_strategy_uses_side_blocks():
    mapper = CoreMapper(aic_cores=20, aiv_cores=40)

    result = mapper.map(
        grid=(128, 32),
        per_block_span_aic_us=11.625,
        per_block_span_aiv_us=28.455,
        task_type="MIX_AIC",
        mix_block_num=8192,
        strategy="mix_block_total",
    )

    assert result.total_blocks == 4096
    assert result.mix_block_num == 8192
    assert result.blocks_aic == 4096
    assert result.blocks_aiv == 8192
    assert result.waves_aic == 205
    assert result.waves_aiv == 205
    assert result.e2e_wall_us == pytest.approx(5833.275)
    assert result.strategy == "mix_block_total"


def test_mix_aic_alternatives_include_auditable_strategies():
    mapper = CoreMapper(aic_cores=20, aiv_cores=40)

    alternatives = mapper.map_alternatives(
        grid=4096,
        per_block_span_aic_us=11.625,
        per_block_span_aiv_us=28.455,
        task_type="MIX_AIC",
        mix_block_num=8192,
    )

    by_strategy = {r.strategy: r for r in alternatives}
    assert set(by_strategy) >= {
        "paired_block",
        "mix_block_total",
        "bottleneck_core_count",
    }
    assert by_strategy["paired_block"].e2e_wall_us == pytest.approx(2930.865)
    assert by_strategy["mix_block_total"].e2e_wall_us == pytest.approx(5833.275)
    assert by_strategy["bottleneck_core_count"].waves_aiv == 205
