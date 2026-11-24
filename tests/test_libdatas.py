from __future__ import annotations

import base64

import numpy as np
import pytest

import libdatas as ld


def test_calc_all_tracks_shapes_and_values() -> None:
    order_data = np.array([1, 2, 1, 3])
    mark, cumulative = ld.calcAllTracks(order_data, 4, 3)
    assert mark.shape == (4, 3)
    assert cumulative.shape == (4, 3)
    assert mark[0].tolist() == [1, 0, 0]
    assert cumulative[-1].tolist() == [2, 1, 1]


def test_calc_all_tracks_rejects_short_window() -> None:
    with pytest.raises(ValueError, match="Длина окна"):
        ld.calcAllTracks(np.array([1, 2, 3]), 5, 20)


def test_calc_all_tracks_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="диапазоне"):
        ld.calcAllTracks(np.array([1, 2, 17, 4, 5]), 5, 10)


def test_calc_passport_histogram() -> None:
    matrix = np.array([[0, 1], [0, 1], [1, 0]])
    passport = ld.calcPassport(matrix, max_deep=2)
    assert passport.tolist() == [[1, 1], [1, 1], [1, 1]]


def test_extract_track_xy_absent_track() -> None:
    separate = np.zeros((5, 3), dtype=np.int32)
    separate[0, 0] = 1
    cumulative = np.cumsum(separate, axis=0, dtype=np.int32)
    track, leap = ld.extract_track_xy(separate, cumulative, 2)
    assert track.shape[0] == 2
    assert leap.shape == (2, 2)


def test_calc_leap_points_shapes() -> None:
    passport = np.array([[1, 0], [0, 1], [1, 0]])
    leap_points = [np.array([[0, 1, 2], [0, 1, 1]])]
    heatmap, zones = ld.calcLeapPoints(passport, leap_points)
    assert heatmap.shape == (2, 3)
    assert len(zones) == 1
    assert zones[0].tolist() == [1]


def test_calc_waves_for_deep_levels() -> None:
    passport = np.array([[2, 1, 0], [0, 1, 2]])
    levels, diff = ld.calcWavesForDeepLevels(passport, 3)
    assert levels.shape == passport.shape
    assert diff.shape == passport.shape
    assert (levels + diff == passport).all()


def test_prepare_wave_fft_data_shapes() -> None:
    levels = np.array([[1, -1], [2, 0], [3, 1]])
    result = ld.prepare_wave_fft_data(levels)
    assert set(result) == {0, 1}
    assert result[0]["fft_y"].shape == (2, 3)


def test_parse_numeric_csv_comma() -> None:
    arr = ld._parse_numeric_csv("1,2\n3,4\n")
    assert arr.tolist() == [[1, 2], [3, 4]]


def test_parse_numeric_csv_single_column() -> None:
    arr = ld._parse_numeric_csv("1\n2\n3\n")
    assert arr.shape == (3, 1)


def test_parse_numeric_csv_rejects_text() -> None:
    with pytest.raises(ValueError, match="числовая матрица"):
        ld._parse_numeric_csv("date,value\n1,2\n")


def test_read_file_content_base64() -> None:
    content = "data:text/csv;base64," + base64.b64encode(b"1\t2\n3\t4\n").decode()
    arr, rows, cols = ld.readFileContent(content, False)
    assert arr.tolist() == [[1, 2], [3, 4]]
    assert (rows, cols) == (2, 2)


def test_read_file_content_rejects_empty() -> None:
    with pytest.raises(ValueError, match="Пустой файл"):
        ld.readFileContent("", False)


def test_match_prognosis_nonzero(csv_4x20: np.ndarray) -> None:
    import pipeline

    core = pipeline.compute_core(csv_4x20[1:201, 0], 200, 20, 0, 0)
    assert sum(core.chances) > 0
    assert core.mr_table.shape[0] == 20


def test_calc_inverse_matrix_shape() -> None:
    order_data = np.array([1, 2, 1, 3])
    matrix_f = ld.calcInverseMatrix(order_data[::-1], 4, 3, 3)
    assert matrix_f.shape == (3, 4)
