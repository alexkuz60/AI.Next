from __future__ import annotations

import numpy as np

import pipeline


def test_calc_all_datas_smoke(csv_4x20: np.ndarray) -> None:
    fig_matrix, fig_chances, levels_store, mr_table, calc_time = pipeline.calcAllDatas(
        csv_4x20[1:201, 0], 200, 20, 0, 0
    )
    assert len(fig_matrix.data) == 7
    assert len(fig_chances.data) == 1
    assert set(levels_store) == {"diff_waves", "matrix_levels", "levels_fft", "leaps_zones"}
    assert mr_table.shape[0] == 20
    assert calc_time >= 0


def test_calc_f_does_not_crash(csv_4x20: np.ndarray) -> None:
    core = pipeline.compute_core(csv_4x20[1:201, 0], 200, 20, 0, 0, calc_f=True)
    assert core.matrix_f.shape == (20, 200)
    fig = pipeline.build_matrix_figure(core, [0.5, 0.5], 1)
    assert len(fig.data) == 8


def test_deep_mode_changes_figure(csv_4x20: np.ndarray) -> None:
    core0 = pipeline.compute_core(csv_4x20[1:201, 0], 200, 20, 0, 0)
    core1 = pipeline.compute_core(csv_4x20[1:201, 0], 200, 20, 0, 1)
    fig0 = pipeline.build_matrix_figure(core0, [0.5, 0.5], 1)
    fig1 = pipeline.build_matrix_figure(core1, [0.5, 0.5], 1)
    assert fig0.to_json() != fig1.to_json()


def test_track_zones_follow_track(csv_4x20: np.ndarray) -> None:
    core0 = pipeline.compute_core(csv_4x20[1:201, 0], 200, 20, 0, 0)
    core5 = pipeline.compute_core(csv_4x20[1:201, 0], 200, 20, 5, 0)
    assert len(core0.leaps_zones[0]) != len(core5.leaps_zones[5])
    fig0 = pipeline.build_matrix_figure(core0, [0.5, 0.5], 1)
    fig5 = pipeline.build_matrix_figure(core5, [0.5, 0.5], 1)
    assert fig0.to_json() != fig5.to_json()


def test_deep_is_clamped(csv_4x20: np.ndarray) -> None:
    core = pipeline.compute_core(csv_4x20[1:201, 0], 200, 20, 0, 0)
    fig = pipeline.build_matrix_figure(core, [0.5, 0.5], 999)
    assert len(fig.data) == 7
