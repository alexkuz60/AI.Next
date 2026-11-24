from __future__ import annotations

import dash_mantine_components as dmc
from dash import Dash
from dash_mantine_components import pre_render_color_scheme

import config
import libdatas as ld
import pipeline
from callbacks import register_callbacks
from layout import build_layout
from state import CalcCache, FileEntry, FileStore


def create_app() -> Dash:
    """Create and configure the Dash application."""
    file_data, matrix_data, matrix_size, max_size, max_order, max_nums, track_num = (
        ld.uploadFileByName(
            str(config.DEFAULT_CSV),
            config.STARTUP_ORDER,
            config.STARTUP_MATRIX_SIZE,
            config.STARTUP_HIST_BACK,
            config.STARTUP_SORTED,
        )
    )

    store = FileStore()
    upload_id = store.put(
        FileEntry(
            filename=str(config.DEFAULT_CSV.relative_to(config.BASE_DIR)),
            data=file_data,
            nums=max_nums,
            orders=max_order,
        )
    )

    order_data = matrix_data[:, config.STARTUP_ORDER]
    core = pipeline.compute_core(
        order_data,
        matrix_size,
        max_nums,
        track_num,
        config.STARTUP_DEEP_MODE,
    )

    calc_cache = CalcCache()
    calc_id = calc_cache.put(
        core,
        key=(
            upload_id,
            config.STARTUP_ORDER,
            config.STARTUP_HIST_BACK,
            matrix_size,
            track_num,
            config.STARTUP_DEEP_MODE,
            False,
            config.STARTUP_SORTED,
        ),
    )

    fig_matrix, fig_chances = pipeline.build_figures(
        core,
        [0.5, 0.5],
        config.STARTUP_DEEP,
        config.STARTUP_DEEP_MODE,
    )

    file_props = {
        "upload_id": upload_id,
        "filename": str(config.DEFAULT_CSV.relative_to(config.BASE_DIR)),
        "nums": max_nums,
        "orders": max_order,
        "deep_mode": config.STARTUP_DEEP_MODE,
        "deep": config.STARTUP_DEEP,
        "sorted": config.STARTUP_SORTED,
    }
    calc_store = {"calc_id": calc_id, "track_num": track_num}

    app = Dash(
        __name__,
        external_stylesheets=dmc.styles.ALL,
        title="Prognosis Ai.Next",
        suppress_callback_exceptions=True,
    )

    app.layout = dmc.MantineProvider(
        [
            pre_render_color_scheme(),
            build_layout(
                file_props=file_props,
                calc_store=calc_store,
                fig_matrix=fig_matrix,
                fig_chances=fig_chances,
                mr_table=core.mr_table,
                filename=file_props["filename"],
                max_size=max_size,
                max_order=max_order,
                max_nums=max_nums,
                track_num=track_num,
                matrix_size=matrix_size,
                hist_back=config.STARTUP_HIST_BACK,
                order=config.STARTUP_ORDER,
                deep=config.STARTUP_DEEP,
                deep_mode=config.STARTUP_DEEP_MODE,
                sorted_flag=config.STARTUP_SORTED,
                calc_time=core.calc_time,
            ),
        ],
        theme={"colorScheme": "dark"},
        forceColorScheme="dark",
    )

    register_callbacks(app, store, calc_cache)
    return app
