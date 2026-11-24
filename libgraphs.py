from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import libcore as lc

if TYPE_CHECKING:
    from numpy.typing import NDArray


def drawDeepLevelWaves(
    w1: NDArray[np.int_],
    w2: NDArray[np.int_],
    fft_y: NDArray[np.floating],
    track_zones: NDArray[np.int_],
    deep: int,
    deep_mode: int,
) -> go.Figure:
    """График волн и FFT для выбранной глубины матрицы."""
    x_max = len(w1)
    diff_deep = lc.matchDiffDeep(deep, deep_mode)

    x_vals = np.arange(x_max)

    fig = go.Figure(
        data=[
            go.Bar(
                y=fft_y[0],
                marker_color="rgba(0,128,255,0.5)",
                name="FFT > 0",
            ),
            go.Bar(
                y=fft_y[1],
                marker_color="rgba(255,0,0,0.5)",
                name="FFT < 0",
            ),
            go.Scatter(
                x=x_vals,
                y=w1[:, diff_deep],
                mode="lines+markers",
                name="w1",
                fill="tozeroy",
                fillcolor="rgba(255,0,127,0.15)",
                line=dict(color="red", shape="spline"),
            ),
            go.Scatter(
                x=x_vals,
                y=w2[:, deep],
                mode="lines+markers",
                name="w2",
                fill="tozeroy",
                fillcolor="rgba(127,0,255,0.15)",
                line=dict(color="cyan", shape="spline"),
            ),
        ],
        layout=go.Layout(
            template="plotly_dark",
            title=f"Depth: {deep}",
            xaxis=dict(
                title="x",
                range=[-1, x_max + 1],
                showgrid=True,
                showspikes=True,
                spikemode="across",
                autorange=False,
            ),
            yaxis=dict(
                title="Wave value",
                showgrid=True,
                showspikes=True,
                spikemode="across",
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            showlegend=False,
            height=500,
            hovermode="x unified",
            clickmode="event+select",
        ),
    )

    y_min, y_max = lc.calcWavesMinMax(w1, w2, deep, deep_mode)
    shapes, annotations = lc.createShapesAndAnnotations(track_zones, y_min, y_max)

    fig.update_yaxes(range=[y_min - 1, y_max + 1])
    fig.update_layout(shapes=shapes, annotations=annotations)

    return fig


def drawChancesBar(chances: list[float], max_num: int, win_num: int) -> go.Figure:
    """Горизонтальная гистограмма вероятностей треков."""
    prognosis = lc.chancesToPercentes(chances)
    nums = list(range(1, max_num + 1))
    fig = go.Figure(layout_template="plotly_dark")

    fig.add_hline(y=win_num + 1)
    fig.add_bar(
        x=prognosis,
        y=nums,
        orientation="h",
    )

    tickvals = list(range(max_num + 1))
    fig.update_layout(
        clickmode="event+select",
        hovermode="x unified",
        hoversort="value ascending",
        height=max_num * 18 + 30,
        margin=dict(t=40, l=10, r=10, b=10),
        xaxis=dict(
            range=[0, 100],
            showgrid=True,
            showspikes=True,
            spikemode="across",
        ),
        yaxis=dict(showgrid=True, tickvals=tickvals),
        modebar=dict(
            remove=[
                "toimage", "pan", "select", "lasso",
                "zoom", "zoomin", "zoomout", "logomark",
            ]
        ),
    )
    return fig


def draw3D(data: NDArray[np.floating]) -> go.Figure:
    """3D-поверхность матрицы событийных глубин."""
    fig = go.Figure()

    fig.add_surface(
        name="Matrix surface",
        z=data,
        opacity=0.75,
        connectgaps=False,
        showscale=True,
        showlegend=True,
    )

    fig.update_traces(
        contours_z=dict(
            show=True,
            usecolormap=True,
            highlightcolor="limegreen",
            project_z=True,
        )
    )

    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis=dict(nticks=6),
            yaxis=dict(nticks=20),
            zaxis=dict(nticks=10),
            camera_eye=dict(x=1, y=0, z=0.4),
            aspectratio=dict(x=0.5, y=1.5, z=0.4),
        ),
    )

    return fig


def drawMatrix(
    data_h: NDArray[np.floating],
    calcF: bool,
    data_f: NDArray[np.floating] | list,
    track: NDArray[np.int_],
    leaps: NDArray[np.floating],
) -> go.Figure:
    """Комбинированная матрица: heatmap + треки + точки переходов."""
    x_max = data_h.shape[1]
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            z=data_h,
            xgap=1,
            ygap=1,
            hoverongaps=False,
            colorscale="Viridis",
            showscale=False,
            showlegend=True,
            name="Matrix-H",
        )
    )

    if calcF and data_f is not None and len(data_f):
        fig.add_trace(
            go.Heatmap(
                z=data_f,
                xgap=1,
                ygap=1,
                hoverongaps=False,
                opacity=0.75,
                showscale=False,
                showlegend=True,
                name="Matrix-F",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=track[0],
            y=track[1],
            mode="lines+markers",
            marker=dict(size=8),
            showlegend=True,
            name="Track #",
        )
    )

    fig.add_trace(
        go.Heatmap(
            z=leaps,
            xgap=1,
            ygap=1,
            hoverongaps=False,
            opacity=0.85,
            colorscale="rainbow",
            showscale=False,
            showlegend=True,
            name="Leaps map",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        clickmode="event+select",
        hovermode="closest",
        margin=dict(t=20, l=20, r=20, b=20),
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            orientation="h",
        ),
        coloraxis_showscale=False,
    )

    axis_cfg = dict(
        showspikes=True,
        spikemode="across",
        showgrid=True,
        tick0=1,
        gridwidth=1,
        gridcolor="LightGrey",
        minor_dtick=0.2,
        minor_gridwidth=0.5,
        minor_gridcolor="rgba(0,0,0,0.1)",
        tickfont=dict(size=14),
    )

    fig.update_xaxes(range=[-1, x_max + 1], **axis_cfg)
    fig.update_yaxes(
        autorange="reversed",
        title_text="Events deep",
        title_font=dict(size=16),
        **axis_cfg,
    )

    return fig


def createCombinedSubplot(
    data_h: NDArray[np.floating],
    calcF: bool,
    data_f: NDArray[np.floating] | list,
    track: NDArray[np.int_],
    track_num: int,
    leaps: NDArray[np.floating],
    levels_fft: dict,
    diff_waves: NDArray[np.int_],
    matrix_waves: NDArray[np.int_],
    track_zones: NDArray[np.int_],
    subplots_balance: list[float],
    deep: int,
    deep_mode: int,
) -> go.Figure:
    """Объединённый subplot: волны/FFT сверху, матрица снизу."""
    x_max = len(diff_waves)
    d = levels_fft[deep]
    fft_y = d["fft_y"]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=subplots_balance,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Events Depth Levels", "Tracks Matrix"],
    )

    # --- 1-я фигура: волны и FFT ---
    deep_fig = drawDeepLevelWaves(
        diff_waves, matrix_waves, fft_y, track_zones, deep, deep_mode
    )
    for trace in deep_fig.data:
        fig.add_trace(trace, row=1, col=1)

    fig.update_layout(
        shapes=list(deep_fig.layout.shapes or []),
        annotations=list(fig.layout.annotations or []) + list(deep_fig.layout.annotations or []),
    )
    fig.update_xaxes(
        title_text="x",
        range=[-1, x_max + 1],
        showgrid=True,
        showspikes=True,
        spikemode="across",
        autorange=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Wave value",
        showgrid=True,
        showspikes=True,
        spikemode="across",
        row=1,
        col=1,
    )

    # --- 2-я фигура: матрица ---
    matrix_fig = drawMatrix(data_h, calcF, data_f, track, leaps)

    for trace in matrix_fig.data:
        fig.add_trace(trace, row=2, col=1)

    fig.add_hline(
        y=deep,
        line_color="yellow",
        line_dash="dot",
        opacity=0.5,
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=900,
        margin=dict(t=40, l=20, r=20, b=40),
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            orientation="h",
        ),
        coloraxis_showscale=False,
    )

    axis_cfg = dict(
        showspikes=True,
        spikemode="across",
        showgrid=True,
        tickfont=dict(size=14),
    )

    fig.update_xaxes(
        range=[-1, x_max + 1],
        row=2,
        col=1,
        rangeslider=dict(visible=True, thickness=0.05),
        **axis_cfg,
    )
    fig.update_yaxes(
        autorange="reversed",
        title_text="Events deep",
        title_font=dict(size=16),
        row=2,
        col=1,
        **axis_cfg,
    )
    fig.update_yaxes(row=1, col=1, **axis_cfg)
    fig.update_xaxes(row=1, col=1, **axis_cfg)

    return fig
