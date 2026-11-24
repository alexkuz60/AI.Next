from __future__ import annotations

import dash_mantine_components as dmc
import pandas as pd
from dash import dcc
from dash_iconify import DashIconify

import libdatas as ld

ICON_NAMES = [
    "streamline:wave-signal-solid",
    "material-symbols:airwave-rounded",
    "tabler:waves-electricity",
]


def mode_icon(mode: int) -> DashIconify:
    """Return a fresh icon component for a deep-shift mode."""
    return DashIconify(icon=ICON_NAMES[int(mode) % len(ICON_NAMES)])


def fileInfoCard(
    filename: str,
    arr_size: int,
    max_cols: int,
    max_num: int,
    calc_time: float,
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
                            dmc.ListItem([dmc.Text(f"Matrix max size: {arr_size}")]),
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


def mrTableData(df: pd.DataFrame) -> dict:
    """Convert the markers DataFrame into a dmc.Table data dict."""
    head = ["track"] + [str(column) for column in df.columns]
    body = [
        [str(index)] + [str(value) for value in row]
        for index, row in zip(df.index, df.values.tolist(), strict=True)
    ]
    return {"head": head, "body": body}


def build_layout(
    *,
    file_props: dict,
    calc_store: dict,
    fig_matrix: object,
    fig_chances: object,
    mr_table: pd.DataFrame,
    filename: str,
    max_size: int,
    max_order: int,
    max_nums: int,
    track_num: int,
    matrix_size: int,
    hist_back: int,
    order: int,
    deep: int,
    deep_mode: int,
    sorted_flag: bool,
    calc_time: float,
) -> dmc.AppShell:
    """Build the full application layout."""
    order_selector = dmc.Select(
        data=[{"label": f"Order {i + 1}", "value": str(i + 1)} for i in range(max_order)],
        value=str(order + 1),
        size="xs",
        leftSection=DashIconify(icon="icon-park-solid:horizontal-tidy-up"),
        w=100,
        allowDeselect=False,
        id="id-order-select",
    )

    switch_sort = dmc.Switch(
        size="xs",
        onLabel=DashIconify(icon="tabler:sort-0-9", width=20),
        radius="md",
        label="Sorted",
        labelPosition="left",
        checked=sorted_flag,
        id="id-switch-sort",
    )

    matrix_size_input = dmc.NumberInput(
        value=matrix_size,
        min=2,
        max=max_size,
        prefix="M-size: ",
        leftSection=DashIconify(icon="healthicons:ui-menu-grid"),
        w=150,
        size="xs",
        id="id-matrix-size",
    )

    history_back = dmc.NumberInput(
        id="id-hist-back",
        min=0,
        max=max_size,
        value=hist_back,
        prefix="H-back: ",
        leftSection=DashIconify(icon="ph:arrow-fat-lines-right-bold"),
        w=120,
        size="xs",
    )

    scan_deep = dmc.NumberInput(
        id="id-markov-deep",
        min=0,
        max=20,
        value=deep,
        prefix="M-deep: ",
        leftSection=DashIconify(icon="gravity-ui:list-timeline"),
        w=120,
        size="xs",
    )

    switch_calc_matrix_f = dmc.Switch(
        size="xs",
        onLabel=DashIconify(icon="fa6-solid:left-right", width=20),
        radius="md",
        label="Calc Matrix-F",
        labelPosition="left",
        checked=False,
        id="id-switch-calc-matrix-f",
    )

    track_selector = dmc.Select(
        data=ld.setTrackList(track_num, max_nums, hist_back),
        value=str(track_num + 1),
        size="xs",
        w=130,
        leftSection=DashIconify(icon="bi:bar-chart-steps"),
        allowDeselect=False,
        id="id-track-select",
    )

    diff_mode_menu = dmc.Menu(
        children=[
            dmc.MenuTarget(
                dmc.ActionIcon(mode_icon(deep_mode), id="id-but-diff-mode"),
                boxWrapperProps="children",
            ),
            dmc.MenuDropdown(
                [
                    dmc.MenuLabel("Deep shift mode:"),
                    dmc.MenuDivider(),
                    dmc.MenuItem(
                        "Stazis",
                        id="id-mode-stazis",
                        leftSection=mode_icon(0),
                    ),
                    dmc.MenuItem(
                        "Deep Y",
                        id="id-mode-deep",
                        leftSection=mode_icon(1),
                    ),
                    dmc.MenuItem(
                        "Smart mode",
                        id="id-mode-smart",
                        leftSection=mode_icon(2),
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
        n_clicks=0,
        id="id-but-view-3d",
    )

    but_view_foot_drawer = dmc.ActionIcon(
        DashIconify(icon="majesticons:table-line"),
        n_clicks=0,
        id="id-but-view-foot-drawer",
    )

    app_header = dmc.AppShellHeader(
        [
            dcc.Store(id="id-file-props-store", data=file_props),
            dcc.Store(id="id-calc-store", data=calc_store),
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
                                                filename,
                                                max_size,
                                                max_order,
                                                max_nums,
                                                calc_time,
                                            )
                                        ],
                                        id="id-info-card",
                                    ),
                                    switch_sort,
                                    order_selector,
                                    history_back,
                                    scan_deep,
                                    matrix_size_input,
                                    track_selector,
                                    diff_mode_menu,
                                    switch_calc_matrix_f,
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

    mr_table_component = dmc.Table(
        id="id-mr-table",
        striped=True,
        highlightOnHover=True,
        withTableBorder=True,
        withColumnBorders=True,
        data=mrTableData(mr_table),
        verticalSpacing=1,
        horizontalSpacing=2,
        fz=10,
    )

    foot_drawer = dmc.Drawer(
        [mr_table_component],
        title="Relations Markers",
        position="bottom",
        size="60%",
        padding="md",
        id="id-foot-drawer",
    )

    error_alert = dmc.Alert(
        id="id-error",
        title="Ошибка",
        color="red",
        children="",
        style={"display": "none"},
    )

    modal_3d = dmc.Modal(
        id="id-3d-modal",
        title="3D Surface",
        size="xl",
        opened=False,
        children=[
            dcc.Graph(id="id-3d-graph", style={"height": "70vh"}),
            dmc.Group(
                [dmc.Button("Закрыть", id="id-3d-close", variant="light")],
                justify="flex-end",
                mt="sm",
            ),
        ],
    )

    app_main = dmc.AppShellMain(
        [
            error_alert,
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
                        [
                            dcc.Loading(
                                dcc.Graph(figure=fig_matrix, id="id-matrix"),
                                type="circle",
                            )
                        ],
                        span=44,
                    ),
                    dmc.GridCol(
                        [
                            dmc.Stack(
                                [
                                    dmc.Text("Prognosis.Next:", fw="bolder"),
                                    dcc.Loading(
                                        dcc.Graph(figure=fig_chances, id="id-chances"),
                                        type="circle",
                                    ),
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
            modal_3d,
        ],
    )

    return dmc.AppShell(
        [app_header, app_main],
        header={"height": "lg"},
    )
