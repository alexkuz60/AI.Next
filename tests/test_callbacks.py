from __future__ import annotations

import base64

import callbacks as cb
from state import CalcCache, FileStore


def _content(text: str) -> str:
    return "data:text/csv;base64," + base64.b64encode(text.encode()).decode()


def test_upload_and_recalc() -> None:
    store = FileStore()
    cache = CalcCache()
    result = cb.handle_upload(
        _content("1\t2\n3\t4\n5\t6\n7\t8\n9\t10\n11\t12\n"),
        "t.csv",
        False,
        {},
        store,
        cache,
    )
    assert len(result) == 14
    assert result[12] == ""
    assert result[13]["display"] == "none"
    props = result[3]
    assert props["nums"] == 12
    assert props["orders"] == 2

    recalc = cb.handle_recalc(
        "id-but-recalc",
        1,
        "1",
        1,
        0,
        0.5,
        "1",
        False,
        True,
        2,
        5,
        props,
        store,
        cache,
    )
    assert len(recalc) == 7
    assert recalc[5] == ""
    assert len(recalc[4]["body"]) == 12

    fig = cb.handle_click({"points": [{"curveNumber": 4, "y": 1}]}, 0.5, recalc[2], cache)
    assert len(fig.data) == 8


def test_upload_rejects_text() -> None:
    store = FileStore()
    cache = CalcCache()
    result = cb.handle_upload(
        _content("date,value\n1,2\n"),
        "bad.csv",
        False,
        {},
        store,
        cache,
    )
    assert len(result) == 14
    assert "Не удалось загрузить файл" in result[12]
    assert result[13]["display"] == "block"


def test_recalc_missing_file() -> None:
    store = FileStore()
    cache = CalcCache()
    result = cb.handle_recalc(
        "id-but-recalc",
        1,
        "1",
        1,
        0,
        0.5,
        "1",
        False,
        False,
        0,
        5,
        {"upload_id": "missing"},
        store,
        cache,
    )
    assert result[5] != ""
    assert result[6]["display"] == "block"


def test_diff_mode_and_drawer() -> None:
    props, icon = cb.handle_diff_mode(1, None, None, {"deep_mode": 0}, "id-mode-deep")
    assert props["deep_mode"] == 1
    assert icon is not None
    assert cb.handle_toggle_drawer(1, False) is True
    assert cb.handle_toggle_drawer(1, True) is False


def test_3d_modal() -> None:
    store = FileStore()
    cache = CalcCache()
    result = cb.handle_upload(
        _content("1\t2\n3\t4\n5\t6\n7\t8\n9\t10\n11\t12\n"),
        "t.csv",
        False,
        {},
        store,
        cache,
    )
    opened, fig = cb.handle_3d(1, result[2], cache)
    assert opened is True
    assert len(fig.data) == 1
    assert cb.handle_close_3d(1) is False
