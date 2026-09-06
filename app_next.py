from __future__ import annotations

import os

import numpy as np
from dash import Dash, dcc, Input, Output, State, callback, no_update, ctx
import dash_mantine_components as dmc
from dash_mantine_components import pre_render_color_scheme
from dash_iconify import DashIconify

import libdatas as ld
import libgraphs as lg

# ── Startup defaults (only used for initial layout; runtime state in dcc.Store)
_STARTUP_ORDER = 0
_STARTUP_HIST_BACK = 1
_STARTUP_MATRIX_SIZE = 200
_STARTUP_DEEP_MODE = 0
_STARTUP_DEEP = 0
_STARTUP_SORTED = False

# ── Debug mode from environment variable (security fix)
_DEBUG = os.getenv("APP_DEBUG", "true").lower() == "true"

# ── Icon pack for deep-shift mode
dm_icon0 = DashIconify(icon="streamline:wave-signal-solid")
dm_icon1 = DashIconify(icon="material-symbols:airwave-rounded")
dm_icon2 = DashIconify(icon="tabler:waves-electricity")
dm_icons_pack = [dm_icon0, dm_icon1, dm_icon2]

# ── Initial data load at startup
file_name = "csv_data/4x20.csv"
file_data, matrix_data, matrix_size, max_size, max_order, max_nums, track_num = ld.uploadFileByName(
    file_name,
    _STARTUP_ORDER,
    _STARTUP_MATRIX_SIZE,
    _STARTUP_HIST_BACK,
    _STARTUP_SORTED,
)

fig_matrix, fig_chances, levels_fft_store, mr_table, calc_time = ld.calcAllDatas(
    matrix_data[:, _STARTUP_ORDER], matrix_size, max_nums, track_num, _STARTUP_DEEP_MODE
)

data_props = {
    "filename": file_name,
    "nums": max_nums,
    "orders": max_order,
    "file_data": file_data,
    "deep_mode": _STARTUP_DEEP_MODE,
    "deep": _STARTUP_DEEP,
}

# ================== LAYOUT =====================

app = Dash(
    external_stylesheets=dmc.styles.ALL,
    title="Prognosis Ai.Next",
    prevent_initial_callbacks=True,
)


# =============== HEADER COMPONENTS =============


def fileInfoCard(
    filename: str, arr_size: int, max_cols: int, max_num: int, calc_time: float
) -> dmc.HoverCard:
    """Information card shown in the header."""
    return dmc.HoverCard(
        width=240,
        withArrow=True,
        position="bottom-start",
        offset=10,
        children=[
            dmc.HoverCardTarget(
                dcc.Upload(
                    dmc.ActionIcon(
                        DashIconify(icon="tabler:file-type-csv"), id="id-load-button"
                    ),
                    id="id-upload",
                ),
            ),
            dmc.HoverCardDropdown(
                [
                    dmc.List(
                        [
                            dmc.ListItem([dmc.Text(f"File name: {filename}")]),
                            dmc.ListItem([dmc.Text(f"Matrix max size: {max_size}")]),
                            dmc.ListItem([dmc.Text(f"Orders: {max_cols}")]),
                            dmc.ListItem([dmc.Text(f"Max tracks: {max_num}")]),
                            dmc.ListItem(
                                [dmc.Text(f"Time calc: {calc_time:.3f} sec", c="red")],
                                icon=DashIconify(
                                    icon="mingcute:time-duration-fill", width=50
                                ),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


orderSelector = dmc.Select(
    data=[{"label": f"Order {i+1}", "value": str(i + 1)} for i in range(max_order)],
    value=str(_STARTUP_ORDER + 1),
    size="xs",
    leftSection=DashIconify(icon="icon-park-solid:horizontal-tidy-up"),
    w=100,
    allowDeselect=False,
    id="id-order-select",
)

switchSort = dmc.Switch(
    size="xs",
    onLabel=DashIconify(icon="tabler:sort-0-9", width=20),
    radius="md",
    label="Sorted",
    labelPosition="left",
    checked=False,
    id="id-switch-sort",
)

matrixSize = dmc.NumberInput(
    value=matrix_size,
    min=2,
    max=max_size,
    prefix="M-size: ",
    leftSection=DashIconify(icon="healthicons:ui-menu-grid"),
    w=150,
    size="xs",
    id="id-matrix-size",
)

historyBack = dmc.NumberInput(
    id="id-hist-back",
    min=0,
    value=1,
    prefix="H-back: ",
    leftSection=DashIconify(icon="ph:arrow-fat-lines-right-bold"),
    w=120,
    size="xs",
)

scanDeep = dmc.NumberInput(
    id="id-markov-deep",
    min=2,
    max=20,
    value=2,
    prefix="M-deep: ",
    leftSection=DashIconify(icon="gravity-ui:list-timeline"),
    w=120,
    size="xs",
)

switchCalkMatrixF = dmc.Switch(
    size="xs",
    onLabel=DashIconify(icon="fa6-solid:left-right", width=20),
    radius="md",
    label="Calc Matrix-F",
    labelPosition="left",
    checked=False,
    id="id-switch-calc-matrix-f",
)

trackSelector = dmc.Select(
    data=ld.setTrackList(track_num, max_nums, _STARTUP_HIST_BACK),
    value=str(track_num + 1),
    size="xs",
    w=130,
    leftSection=DashIconify(icon="bi:bar-chart-steps"),
    allowDeselect=False,
    id="id-track-select",
    persistence=True,
)

diffModeMenu = dmc.Menu(
    children=[
        dmc.MenuTarget(
            dmc.ActionIcon(dm_icons_pack[_STARTUP_DEEP_MODE], id="id-but-diff-mode"),
            boxWrapperProps="children",
        ),
        dmc.MenuDropdown(
            [
                dmc.MenuLabel("Deep shift mode:"),
                dmc.MenuDivider(),
                dmc.MenuItem(
                    "Stazis",
                    id="id-mode-stazis",
                    leftSection=dm_icons_pack[0],
                ),
                dmc.MenuItem(
                    "Deep Y",
                    id="id-mode-deep",
                    leftSection=dm_icons_pack[1],
                ),
                dmc.MenuItem(
                    "Smart mode",
                    id="id-mode-smart",
                    leftSection=dm_icons_pack[2],
                    color="yellow",
                ),
            ]
        ),
    ],
    id="id-diff-mode",
    trigger="hover",
    withArrow=True,
)

but_recalc = dmc.ActionIcon(
    DashIconify(icon="fluent:calculator-arrow-clockwise-20-regular"),
    color="green",
    variant="filled",
    n_clicks=0,
    id="id-but-recalc",
)

but_view_3d = dmc.ActionIcon(
    DashIconify(icon="mage:box-3d-scan"),
    color="green",
    variant="filled",
    id="id-but-view-3d",
)

but_view_foot_drawer = dmc.ActionIcon(
    DashIconify(icon="majesticons:table-line"),
    id="id-but-view-foot-drawer",
)

app_header = dmc.AppShellHeader(
    [
        dcc.Store(id="id-file-props-store", data=data_props),
        dmc.Grid(
            [
                dmc.GridCol(
                    [
                        dmc.Group(
                            [
                                dmc.Text(
                                    "XaosLab",
                                    size="xl",
                                    fw="bolder",
                                    variant="gradient",
                                    gradient={"from": "indigo", "to": "yellow", "deg": 45},
                                ),
                                dmc.Box(
                                    [
                                        fileInfoCard(
                                            file_name,
                                            matrix_size,
                                            max_order,
                                            max_nums,
                                            calc_time,
                                        )
                                    ],
                                    id="id-info-card",
                                ),
                                switchSort,
                                orderSelector,
                                historyBack,
                                scanDeep,
                                matrixSize,
                                trackSelector,
                                diffModeMenu,
                                switchCalkMatrixF,
                                but_recalc,
                            ]
                        ),
                    ],
                    span="auto",
                ),
                dmc.GridCol(
                    [dmc.Group([but_view_foot_drawer, but_view_3d], justify="flex-end")],
                    span="content",
                    offset=2,
                ),
            ],
        ),
    ],
    mx=10,
)

excluderMenu = dmc.Menu(
    [
        dmc.MenuTarget(
            dmc.ActionIcon(
                DashIconify(
                    icon="carbon:skill-level-intermediate", rotate=1, flip="horizontal"
                ),
            )
        ),
        dmc.MenuDropdown(
            [
                dmc.MenuLabel("Exclude:"),
                dmc.MenuItem(
                    "Selected", leftSection=DashIconify(icon="tabler:settings")
                ),
                dmc.MenuItem(
                    "Unselected", leftSection=DashIconify(icon="tabler:message")
                ),
                dmc.MenuItem("Save", leftSection=DashIconify(icon="tabler:photo")),
                dmc.MenuDivider(),
                dmc.MenuLabel("Danger Zone"),
                dmc.MenuItem(
                    "Restore last",
                    leftSection=DashIconify(icon="tabler:arrows-left-right"),
                ),
                dmc.MenuItem(
                    "Restore all",
                    leftSection=DashIconify(icon="tabler:trash"),
                    color="red",
                ),
            ]
        ),
    ],
    trigger="hover",
)

mr_table = dmc.Table(
    id="id-mr-table",
    striped=True,
    highlightOnHover=True,
    withTableBorder=True,
    withColumnBorders=True,
    data={"head": mr_table.columns.tolist(), "body": mr_table.values.tolist()},
    verticalSpacing=1,
    horizontalSpacing=2,
    fz=10,
)

foot_drawer = dmc.Drawer(
    [mr_table],
    title="Relations Markers",
    position="bottom",
    size="60%",
    padding="md",
    id="id-foot-drawer",
)

app_main = dmc.AppShellMain(
    [
        dcc.Store(id="id-matrix-store", data=fig_matrix),
        dcc.Store(id="id-levels-store", data=levels_fft_store),
        dmc.Grid(
            [
                dmc.GridCol(
                    [
                        dcc.Slider(
                            id="id-slider-balance",
                            min=0.1,
                            step=0.05,
                            value=0.5,
                            max=1.0,
                            vertical=True,
                            verticalHeight=920,
                            updatemode="mouseup",
                            persistence=True,
                        )
                    ],
                    span=1,
                    mt=0,
                ),
                dmc.GridCol(
                    [dcc.Graph(figure=fig_matrix, id="id-matrix")],
                    span=44,
                ),
                dmc.GridCol(
                    [
                        dmc.Stack(
                            [
                                dmc.Text("Prognosis.Next:", fw="bolder"),
                                excluderMenu,
                                dcc.Graph(figure=fig_chances, id="id-chances"),
                            ],
                            align="left",
                            gap="5",
                        )
                    ],
                    span=5,
                ),
            ],
            columns=50,
            overflow="scroll",
            gutter=5,
            ml=5,
            mt=45,
        ),
        foot_drawer,
    ],
)

layout = dmc.AppShell(
    [app_header, app_main],
    header={"height": "lg"},
)

# ── Global dark-mode CSS injected into <head>
app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            html, body, #_dash-app-content {
                background-color: #141517 !important;
                color: #C1C2C5 !important;
            }
            .mantine-AppShell-header,
            .mantine-Grid,
            .mantine-Stack,
            .mantine-Group {
                background-color: #141517 !important;
            }
            .js-plotly-plot .plotly .bg,
            .js-plotly-plot .plotly .main-svg {
                background: transparent !important;
            }
            .mantine-active { background-color: #141517 !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        {%config%}
        {%scripts%}
        {%renderer%}
    </body>
</html>"""

app.layout = dmc.MantineProvider(
    [
        pre_render_color_scheme(),
        layout,
    ],
    theme={"colorScheme": "dark"},
    forceColorScheme="dark",
)

# ==================================== CALLBACKS =============================


@callback(
    Output("id-matrix", "figure", allow_duplicate=True),
    Output("id-chances", "figure", allow_duplicate=True),
    Output("id-levels-store", "data", allow_duplicate=True),
    Output("id-file-props-store", "data", allow_duplicate=True),
    Output("id-hist-back", "value"),
    Output("id-matrix-size", "max"),
    Output("id-matrix-size", "value"),
    Output("id-order-select", "data"),
    Output("id-order-select", "value"),
    Output("id-track-select", "data"),
    Output("id-track-select", "value"),
    Output("id-info-card", "children"),
    Input("id-upload", "contents"),
    State("id-switch-sort", "checked"),
    State("id-upload", "filename"),
    prevent_initial_call=True,
)
def uploadFile(
    content: str | None, sorted: bool, filename: str | None
) -> tuple:
    """Handle file upload: parse CSV and recalculate all data."""
    if content is None or filename is None:
        return (no_update,) * 12

    csv_arr, matrix_max, max_order = ld.readFileContent(content, sorted)
    order = 0
    orders_list = [
        {"label": f"Order {i+1}", "value": str(i + 1)} for i in range(max_order)
    ]
    h_back = 1
    deep = 0

    max_nums = int(max(csv_arr[:, order]))
    tracks_list = [
        {"label": f"Track {i+1}", "value": str(i + 1)} for i in range(max_nums)
    ]

    track_num = ld.autoSelectNextTrack(csv_arr, h_back, order)

    matrix_size = 250
    if matrix_size > matrix_max:
        matrix_size = matrix_max - h_back

    matrix_graf, chances_bar, levels_store, mr_table, calc_time = ld.calcAllDatas(
        csv_arr[h_back : matrix_size + h_back, order],
        matrix_size,
        max_nums,
        track_num,
        deep_mode=0,
        subplots_balance=[0.5, 0.5],
        calc_f=False,
    )

    # FIX: use uploaded filename instead of hardcoded default
    file_props = {
        "filename": filename,
        "nums": max_nums,
        "orders": max_order,
        "file_data": csv_arr,
        "deep_mode": deep_mode,
        "deep": deep,
    }
    file_info_card = fileInfoCard(filename, matrix_max, max_order, max_nums, calc_time)
    return (
        matrix_graf,
        chances_bar,
        levels_store,
        file_props,
        h_back,
        matrix_max,
        matrix_size,
        orders_list,
        str(order + 1),
        tracks_list,
        str(track_num + 1),
        file_info_card,
    )


@callback(
    Output("id-file-props-store", "data", allow_duplicate=True),
    Output("id-but-diff-mode", "children"),
    Input("id-mode-stazis", "n_clicks"),
    Input("id-mode-deep", "n_clicks"),
    Input("id-mode-smart", "n_clicks"),
    State("id-file-props-store", "data"),
    prevent_initial_call=True,
)
def setDiffMode(mode0: int, mode1: int, mode2: int, file_props: dict) -> tuple:
    """Switch deep-shift analysis mode (Stazis / Deep Y / Smart)."""
    if ctx.triggered_id == "id-mode-stazis":
        deep_mode = 0
    elif ctx.triggered_id == "id-mode-deep":
        deep_mode = 1
    elif ctx.triggered_id == "id-mode-smart":
        deep_mode = 2
    else:
        deep_mode = file_props.get("deep_mode", 0)

    file_props["deep_mode"] = deep_mode
    but_icon = dmc.ActionIcon(dm_icons_pack[deep_mode])
    return file_props, but_icon


@callback(
    Output("id-matrix", "figure", allow_duplicate=True),
    Output("id-chances", "figure", allow_duplicate=True),
    Output("id-levels-store", "data", allow_duplicate=True),
    Output("id-track-select", "value", allow_duplicate=True),
    Input("id-but-recalc", "n_clicks"),
    Input("id-order-select", "value"),
    Input("id-hist-back", "value"),
    Input("id-matrix-size", "n_blur"),
    Input("id-slider-balance", "value"),
    State("id-matrix-size", "value"),
    State("id-file-props-store", "data"),
    State("id-track-select", "value"),
    State("id-switch-calc-matrix-f", "checked"),
    prevent_initial_call=True,
)
def recalcMatrix(
    _,
    sel_order: str,
    hist_back: int,
    n_blur: int,
    slider_val: float,
    m_size: str,
    file_props: dict,
    sel_track: str,
    calc_f: bool,
) -> tuple:
    """Recalculate matrix and all charts when parameters change."""
    order = int(sel_order) - 1
    file_data = np.array(file_props["file_data"])
    order_data = file_data[hist_back : int(m_size) + hist_back, order]
    max_nums = file_props["nums"]
    deep_mode = file_props["deep_mode"]

    track_num = int(sel_track) - 1
    if ctx.triggered_id == "id-order-select" or (
        ctx.triggered_id == "id-hist-back" and hist_back > 0
    ):
        track_num = ld.autoSelectNextTrack(file_data, hist_back, order)
    track_val = str(track_num + 1)

    subplots_balance = [1.0 - slider_val, slider_val]

    matrix_fig, chances_fig, levels_fft_store, mr_table, time_recalc = ld.calcAllDatas(
        order_data,
        int(m_size),
        int(max_nums),
        track_num,
        deep_mode,
        subplots_balance,
        calc_f=calc_f or False,
    )
    return matrix_fig, chances_fig, levels_fft_store, track_val


@callback(
    Output("id-matrix", "figure", allow_duplicate=True),
    Input("id-matrix", "clickData"),
    State("id-matrix", "figure"),
    State("id-track-select", "value"),
    State("id-levels-store", "data"),
    State("id-file-props-store", "data"),
    prevent_initial_call=True,
)
def updateLevelOnClickMatrix(
    clickData: dict,
    matrix: dict,
    track_num: str,
    levels_fft_store: dict,
    file_props: dict,
) -> dict | type(no_update):
    """Update charts when user clicks on a matrix depth level."""
    if not clickData or "points" not in clickData:
        return no_update

    point = clickData["points"][0]
    y = int(point["y"])
    idx_subplot = point["curveNumber"]

    if idx_subplot > 3:
        deep = str(y)

        idx_last_shape = len(matrix["layout"]["shapes"]) - 1
        deep_shape = matrix["layout"]["shapes"][idx_last_shape]

        levels_fft = levels_fft_store["levels_fft"]
        level = levels_fft[deep]
        fft_y = level["fft_y"]

        deep_mode = file_props["deep_mode"]
        diffs_w1 = np.array(levels_fft_store["diff_waves"], dtype=int)
        levels_w2 = np.array(levels_fft_store["matrix_levels"], dtype=int)
        y_min, y_max = ld.calcWavesMinMax(diffs_w1, levels_w2, y, deep_mode)

        diff_wave = diffs_w1[:, y]
        matrix_level = levels_w2[:, y]

        matrix["data"][0]["y"] = fft_y[0]
        matrix["data"][1]["y"] = fft_y[1]
        matrix["data"][2]["y"] = diff_wave
        matrix["data"][3]["y"] = matrix_level

        tr_num = int(track_num) - 1
        tr_zones = levels_fft_store["leaps_zones"][tr_num]

        shapes, annotations = ld.createShapesAndAnnotations(tr_zones, y_min, y_max)
        matrix["layout"]["shapes"] = shapes
        matrix["layout"]["annotations"] = annotations

        deep_shape["y0"] = y
        deep_shape["y1"] = y
        matrix["layout"]["shapes"].append(deep_shape)

        return matrix
    else:
        return no_update


@callback(
    Output("id-foot-drawer", "opened"),
    Input("id-but-view-foot-drawer", "n_clicks"),
    State("id-foot-drawer", "opened"),
    prevent_initial_call=True,
)
def toggleFootDrawer(n_clicks: int | None, is_open: bool) -> bool:
    """Toggle the foot drawer open/closed."""
    if n_clicks is None:
        return False
    return not is_open


@callback(
    Input("id-but-view-3d", "n_clicks"),
    State("id-levels-store", "data"),
    prevent_initial_call=True,
)
def view3dSurface(n_clicks: int | None, levels: dict) -> None:
    """Open 3D surface plot in a new browser window."""
    if n_clicks is None or levels is None:
        return
    data = np.array(levels["diff_waves"])
    fig_3d = lg.draw3D(data)
    fig_3d.show()


if __name__ == "__main__":
    app.run(debug=_DEBUG)
