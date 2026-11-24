from __future__ import annotations

import numpy as np

import libgraphs as lg


def test_draw_matrix_with_calc_f() -> None:
    data_h = np.ones((3, 4))
    data_f = np.ones((3, 4))
    track = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    leaps = np.ones((3, 4))
    fig = lg.drawMatrix(data_h, True, data_f, track, leaps)
    assert len(fig.data) == 4
    assert fig.data[1].name == "Matrix-F"


def test_draw_matrix_without_calc_f() -> None:
    data_h = np.ones((3, 4))
    track = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    leaps = np.ones((3, 4))
    fig = lg.drawMatrix(data_h, False, [], track, leaps)
    assert len(fig.data) == 3


def test_create_combined_subplot_keeps_titles() -> None:
    rows, deep = 6, 3
    data_h = np.ones((deep, rows))
    diff_waves = np.ones((rows, deep), dtype=int)
    matrix_levels = np.ones((rows, deep), dtype=int)
    levels_fft = {
        i: {"fft_y": np.zeros((2, rows))} for i in range(deep)
    }
    track = np.array([[0, rows - 1], [0, 0]])
    fig = lg.createCombinedSubplot(
        data_h,
        False,
        [],
        track,
        0,
        np.ones((deep, rows)),
        levels_fft,
        diff_waves,
        matrix_levels,
        np.array([2]),
        [0.5, 0.5],
        1,
        0,
    )
    assert len(fig.data) == 7
    titles = [annotation.text for annotation in fig.layout.annotations]
    assert "Events Depth Levels" in titles
    assert "Tracks Matrix" in titles


def test_draw_chances_bar() -> None:
    fig = lg.drawChancesBar([1, 2, 3], 3, 1)
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [1, 2, 3]
