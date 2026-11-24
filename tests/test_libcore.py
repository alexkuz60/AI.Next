from __future__ import annotations

import numpy as np
import pytest

import libcore as lc


def test_match_diff_deep() -> None:
    assert lc.matchDiffDeep(0, 0) == 0
    assert lc.matchDiffDeep(3, 0) == 3
    assert lc.matchDiffDeep(3, 1) == 2
    assert lc.matchDiffDeep(0, 1) == 0
    assert lc.matchDiffDeep(3, 2) == 3


def test_chances_to_percentes_sums_to_100() -> None:
    result = lc.chancesToPercentes([1, 1, 1])
    assert result.sum() == pytest.approx(100.0, abs=0.05)
    assert np.all(result >= 0)


def test_chances_to_percentes_all_zero() -> None:
    result = lc.chancesToPercentes([0, 0, 0])
    assert np.all(result == 0)


def test_create_shapes_and_annotations() -> None:
    shapes, annotations = lc.createShapesAndAnnotations(np.array([2, 5]), -1.0, 3.0)
    assert len(shapes) == 4
    assert len(annotations) == 2
    assert annotations[0]["text"] == "#1:x2"


def test_format_markers_map_to_table() -> None:
    markers = {"M0": [{"track": 0, "count": 1, "pos": [1]}]}
    table = lc.format_markers_map_to_table(markers, max_track=3)
    assert table.shape == (4, 1)
    assert table.at[0, "M0"] == [1]
    assert table.at[3, "M0"] == ""
