from __future__ import annotations

import numpy as np
from dash import Dash, Input, Output, State, ctx, no_update

import config
import libdatas as ld
import libgraphs as lg
import pipeline
from layout import fileInfoCard, mode_icon, mrTableData
from state import CalcCache, FileEntry, FileStore

NO_UPDATE_14 = (no_update,) * 14
NO_UPDATE_7 = (no_update,) * 7


def _error(message: str) -> tuple[str, dict]:
    return message, {"display": "block"}


def _ok() -> tuple[str, dict]:
    return "", {"display": "none"}


def handle_upload(
    contents: str | None,
    filename: str | None,
    sorted_flag: bool | None,
    file_props: dict | None,
    store: FileStore,
    calc_cache: CalcCache,
) -> tuple:
    """Parse an uploaded CSV, store it server-side and rebuild all figures."""
    if not contents or not filename:
        return NO_UPDATE_14

    try:
        csv_arr, matrix_max, max_order = ld.readFileContent(contents, bool(sorted_flag))
        h_back = config.STARTUP_HIST_BACK
        matrix_size = min(config.STARTUP_MATRIX_SIZE, matrix_max - h_back)
        if matrix_size < 2:
            raise ValueError("В файле слишком мало строк для матрицы")

        max_nums = int(csv_arr.max())
        if max_nums < 1:
            raise ValueError("В файле нет положительных номеров треков")

        order = 0
        track_num = ld.autoSelectNextTrack(csv_arr, h_back, order)
        track_num = max(0, min(track_num, max_nums - 1))

        deep_mode = int((file_props or {}).get("deep_mode", 0))
        deep = max(0, int((file_props or {}).get("deep", 0)))

        upload_id = store.put(
            FileEntry(filename=filename, data=csv_arr, nums=max_nums, orders=max_order)
        )
        order_data = csv_arr[h_back: h_back + matrix_size, order]
        core = pipeline.compute_core(
            order_data, matrix_size, max_nums, track_num, deep_mode
        )
        key = (
            upload_id,
            order,
            h_back,
            matrix_size,
            track_num,
            deep_mode,
            False,
            bool(sorted_flag),
        )
        calc_id = calc_cache.put(core, key=key)
        fig_matrix, fig_chances = pipeline.build_figures(
            core, [0.5, 0.5], deep, deep_mode
        )

        props = {
            "upload_id": upload_id,
            "filename": filename,
            "nums": max_nums,
            "orders": max_order,
            "deep_mode": deep_mode,
            "deep": deep,
            "sorted": bool(sorted_flag),
        }
        orders_list = [
            {"label": f"Order {i + 1}", "value": str(i + 1)} for i in range(max_order)
        ]
        error_text, error_style = _ok()
        return (
            fig_matrix,
            fig_chances,
            {"calc_id": calc_id, "track_num": track_num},
            props,
            orders_list,
            "1",
            ld.setTrackList(track_num, max_nums, h_back),
            str(track_num + 1),
            fileInfoCard(filename, matrix_max, max_order, max_nums, core.calc_time),
            matrix_max - h_back,
            matrix_size,
            mrTableData(core.mr_table),
            error_text,
            error_style,
        )
    except Exception as exc:
        error_text, error_style = _error(f"Не удалось загрузить файл: {exc}")
        return (no_update,) * 12 + (error_text, error_style)


def handle_recalc(
    triggered_id: str | None,
    _n_clicks: int | None,
    sel_order: str | None,
    hist_back: int | None,
    _n_blur: int | None,
    slider_val: float | None,
    sel_track: str | None,
    sorted_flag: bool | None,
    calc_f: bool | None,
    deep: int | None,
    m_size: int | None,
    file_props: dict | None,
    store: FileStore,
    calc_cache: CalcCache,
) -> tuple:
    """Recalculate figures when any matrix parameter changes."""
    if not triggered_id or not file_props:
        return NO_UPDATE_7

    entry = store.get(file_props.get("upload_id"))
    if entry is None:
        error_text, error_style = _error("Файл не найден, загрузите CSV заново")
        return (no_update,) * 5 + (error_text, error_style)

    try:
        rows = entry.data.shape[0]
        h_back = max(0, int(hist_back or 0))
        h_back = min(h_back, max(0, rows - 2))
        m_size = max(2, min(int(m_size or 0), rows - h_back))

        order = int(sel_order or 1) - 1
        if not 0 <= order < entry.orders:
            raise ValueError("Некорректный номер порядка")

        track_num = int(sel_track or 1) - 1
        track_num = max(0, min(track_num, entry.nums - 1))
        deep = max(0, int(deep or 0))
        deep_mode = int(file_props.get("deep_mode", 0))

        data = np.sort(entry.data, axis=1) if sorted_flag else entry.data
        order_data = data[h_back: h_back + m_size, order]

        key = (
            file_props.get("upload_id"),
            order,
            h_back,
            m_size,
            track_num,
            deep_mode,
            bool(calc_f),
            bool(sorted_flag),
        )
        core = calc_cache.find(key)
        if core is None:
            core = pipeline.compute_core(
                order_data, m_size, entry.nums, track_num, deep_mode, bool(calc_f)
            )
        calc_id = calc_cache.put(core, key=key)

        balance = min(1.0, max(0.1, float(slider_val or 0.5)))
        fig_matrix, fig_chances = pipeline.build_figures(
            core, [1.0 - balance, balance], deep, deep_mode
        )

        error_text, error_style = _ok()
        return (
            fig_matrix,
            fig_chances,
            {"calc_id": calc_id, "track_num": track_num},
            fileInfoCard(
                entry.filename, rows, entry.orders, entry.nums, core.calc_time
            ),
            mrTableData(core.mr_table),
            error_text,
            error_style,
        )
    except Exception as exc:
        error_text, error_style = _error(f"Ошибка пересчёта: {exc}")
        return (no_update,) * 5 + (error_text, error_style)


def handle_click(
    click_data: dict | None,
    slider_val: float | None,
    calc_store: dict | None,
    calc_cache: CalcCache,
) -> object:
    """Rebuild the matrix figure for the depth level clicked by the user."""
    if not click_data or "points" not in click_data or not calc_store:
        return no_update

    point = click_data["points"][0]
    if int(point.get("curveNumber", 0)) <= 3:
        return no_update

    core = calc_cache.get(calc_store.get("calc_id"))
    if core is None:
        return no_update

    deep = int(point.get("y", 0))
    if not 0 <= deep < len(core.levels_fft):
        return no_update

    balance = min(1.0, max(0.1, float(slider_val or 0.5)))
    return pipeline.build_matrix_figure(core, [1.0 - balance, balance], deep)


def handle_diff_mode(
    _mode0: int | None,
    _mode1: int | None,
    _mode2: int | None,
    file_props: dict | None,
    triggered_id: str | None,
) -> tuple:
    """Switch deep-shift analysis mode (Stazis / Deep Y / Smart)."""
    if not file_props:
        return no_update, no_update

    mapping = {"id-mode-stazis": 0, "id-mode-deep": 1, "id-mode-smart": 2}
    deep_mode = mapping.get(triggered_id, int(file_props.get("deep_mode", 0)))
    props = {**file_props, "deep_mode": deep_mode}
    return props, mode_icon(deep_mode)


def handle_toggle_drawer(n_clicks: int | None, is_open: bool | None) -> bool:
    """Toggle the foot drawer open/closed."""
    if not n_clicks:
        return False
    return not bool(is_open)


def handle_3d(
    n_clicks: int | None,
    calc_store: dict | None,
    calc_cache: CalcCache,
) -> tuple:
    """Open the 3D surface in a browser modal."""
    if not n_clicks or not calc_store:
        return no_update, no_update

    core = calc_cache.get(calc_store.get("calc_id"))
    if core is None:
        return no_update, no_update

    return True, lg.draw3D(np.array(core.diff_waves))


def handle_close_3d(_n_clicks: int | None) -> bool:
    return False


def register_callbacks(app: Dash, store: FileStore, calc_cache: CalcCache) -> None:
    """Register all Dash callbacks on the application instance."""

    @app.callback(
        Output("id-matrix", "figure", allow_duplicate=True),
        Output("id-chances", "figure", allow_duplicate=True),
        Output("id-calc-store", "data", allow_duplicate=True),
        Output("id-file-props-store", "data", allow_duplicate=True),
        Output("id-order-select", "data"),
        Output("id-order-select", "value"),
        Output("id-track-select", "data"),
        Output("id-track-select", "value"),
        Output("id-info-card", "children", allow_duplicate=True),
        Output("id-matrix-size", "max"),
        Output("id-matrix-size", "value"),
        Output("id-mr-table", "data", allow_duplicate=True),
        Output("id-error", "children", allow_duplicate=True),
        Output("id-error", "style", allow_duplicate=True),
        Input("id-upload", "contents"),
        State("id-upload", "filename"),
        State("id-switch-sort", "checked"),
        State("id-file-props-store", "data"),
        prevent_initial_call=True,
    )
    def upload_file(contents, filename, sorted_flag, file_props):
        return handle_upload(contents, filename, sorted_flag, file_props, store, calc_cache)

    @app.callback(
        Output("id-matrix", "figure", allow_duplicate=True),
        Output("id-chances", "figure", allow_duplicate=True),
        Output("id-calc-store", "data", allow_duplicate=True),
        Output("id-info-card", "children", allow_duplicate=True),
        Output("id-mr-table", "data", allow_duplicate=True),
        Output("id-error", "children", allow_duplicate=True),
        Output("id-error", "style", allow_duplicate=True),
        Input("id-but-recalc", "n_clicks"),
        Input("id-order-select", "value"),
        Input("id-hist-back", "value"),
        Input("id-matrix-size", "n_blur"),
        Input("id-slider-balance", "value"),
        Input("id-track-select", "value"),
        Input("id-switch-sort", "checked"),
        Input("id-switch-calc-matrix-f", "checked"),
        Input("id-markov-deep", "value"),
        State("id-matrix-size", "value"),
        State("id-file-props-store", "data"),
        prevent_initial_call=True,
    )
    def recalc_matrix(
        n_clicks,
        sel_order,
        hist_back,
        n_blur,
        slider_val,
        sel_track,
        sorted_flag,
        calc_f,
        deep,
        m_size,
        file_props,
    ):
        return handle_recalc(
            ctx.triggered_id,
            n_clicks,
            sel_order,
            hist_back,
            n_blur,
            slider_val,
            sel_track,
            sorted_flag,
            calc_f,
            deep,
            m_size,
            file_props,
            store,
            calc_cache,
        )

    @app.callback(
        Output("id-matrix", "figure", allow_duplicate=True),
        Input("id-matrix", "clickData"),
        Input("id-slider-balance", "value"),
        State("id-calc-store", "data"),
        prevent_initial_call=True,
    )
    def update_level_on_click(click_data, slider_val, calc_store):
        return handle_click(click_data, slider_val, calc_store, calc_cache)

    @app.callback(
        Output("id-file-props-store", "data", allow_duplicate=True),
        Output("id-but-diff-mode", "children"),
        Input("id-mode-stazis", "n_clicks"),
        Input("id-mode-deep", "n_clicks"),
        Input("id-mode-smart", "n_clicks"),
        State("id-file-props-store", "data"),
        prevent_initial_call=True,
    )
    def set_diff_mode(mode0, mode1, mode2, file_props):
        return handle_diff_mode(mode0, mode1, mode2, file_props, ctx.triggered_id)

    @app.callback(
        Output("id-foot-drawer", "opened"),
        Input("id-but-view-foot-drawer", "n_clicks"),
        State("id-foot-drawer", "opened"),
        prevent_initial_call=True,
    )
    def toggle_foot_drawer(n_clicks, is_open):
        return handle_toggle_drawer(n_clicks, is_open)

    @app.callback(
        Output("id-3d-modal", "opened", allow_duplicate=True),
        Output("id-3d-graph", "figure"),
        Input("id-but-view-3d", "n_clicks"),
        State("id-calc-store", "data"),
        prevent_initial_call=True,
    )
    def view_3d_surface(n_clicks, calc_store):
        return handle_3d(n_clicks, calc_store, calc_cache)

    @app.callback(
        Output("id-3d-modal", "opened", allow_duplicate=True),
        Input("id-3d-close", "n_clicks"),
        prevent_initial_call=True,
    )
    def close_3d_modal(n_clicks):
        return handle_close_3d(n_clicks)
