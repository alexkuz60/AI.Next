import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import libdatas as ld

def drawDeepLevelWaves(w1, w2, fft_y, track_zones, deep, deep_mode):
    x_max = len(w1)
    diff_deep = ld.matchDiffDeep(deep, deep_mode)

    x_vals = np.arange(x_max)

    # Используем go.Figure с data и layout сразу для компактности
    fig = go.Figure(
        data=[
            # Bar for positive FFT
            go.Bar(
                y=fft_y[0],
                marker_color='rgba(0,128,255,0.5)',
                name='FFT > 0'
            ),
            # Bar for negative FFT
            go.Bar(
                y=fft_y[1],
                marker_color='rgba(255,0,0,0.5)',
                name='FFT < 0'
            ),
            go.Scatter(
                x=x_vals, y=w1[:, diff_deep],
                mode='lines+markers',
                name='w1',
                fill='tozeroy',
                fillcolor="rgba(255,0,127,0.15)",
                line=dict(color='red', shape='spline')
            ),
            go.Scatter(
                x=x_vals, y=w2[:, deep],
                mode='lines+markers',
                name='w2',
                fill='tozeroy',
                fillcolor="rgba(127,0,255,0.15)",
                line=dict(color='cyan', shape='spline')
            )
        ],
        layout=go.Layout(
            template='plotly_dark',
            title=f"Depth: {deep}",
            xaxis=dict(
                title="x",
                range=[-1, x_max+1],
                showgrid=True,
                showspikes=True,
                spikemode='across',
                autorange=False
            ),
            yaxis=dict(
                title="Waveы value",
                showgrid=True,
                showspikes=True,
                spikemode='across'
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            showlegend=False,
            height=500
        )
    )
    #----------------Метки точек перехода для трека------------------
    # Оптимизируем вычисление y_min и y_max для корректного диапазона
    #y_min = min(w1[:, deep].min(), w2[:, diff_deep].min()) - 1
    #y_max = max(w1[:, deep].max(), w2[:, diff_deep].max()) + 1
    y_min, y_max = ld.calcWavesMinMax(w1, w2, deep, deep_mode)
    shapes, annotations = ld.createShapesAndAnnotations(track_zones, y_min, y_max)
    
    fig.update_yaxes(range=[y_min - 1, y_max + 1])
    fig.update_layout(shapes=shapes, annotations=annotations)

    return fig
#-------------------------------------------------------------------------------

def drawChancesBar(chances, max_num, win_num):
    prognosis = ld.chancesToPercentes(chances)
    #print(prognosis)
    nums = list(range(1, max_num + 1))   #np.arange(1, max, dtype=int)
    fig = go.Figure(layout_template="plotly_dark")

    fig.add_hline(
        y=win_num+1,
    )
    fig.add_bar(
        x=prognosis,
        y=nums,
        orientation='h',
    )
    
    tickvals = list(range(max_num+1))
    fig.update_layout(
        #title_text="Events chances",
        clickmode='event+select',
        height = max_num * 18 + 20 + 10,
        margin = dict(t=40, l=10, r=10, b=10),
        xaxis = dict(range=[0, 100], showgrid=True, showspikes=True, spikemode='across'),
        yaxis = dict(showgrid = True, tickvals=tickvals),
        modebar = dict(remove = ["toimage", "pan", "select", "lasso", "zoom", "zoomin", "zoomout", "logomark"])
    )
    return fig
#--------------------------------------------------------------

def draw3D(matrix_h):
    fig = go.Figure()
    
    fig.add_surface(
        name="Matrix surface",
        z = matrix_h,
        opacity = 0.75,
        connectgaps = False,
        #colorscale='Viridis',
        showscale = True,
        showlegend = True,
    )
    """
    fig.add_scatter3d(
        name="Tracks",
        z=matrix_h,
        opacity=0.85,
        showlegend=True,
        mode='lines+markers',
        line_width=5,
        #line_color='cyan',
    )
    """
    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
    
    fig.update_layout(
        template="plotly_dark",
        scene = {
            "xaxis": {"nticks": 6},
            "yaxis": {"nticks": 20},
            "zaxis": {"nticks": 10},
            'camera_eye': {"x": 1, "y": 0, "z": 0.4},
            "aspectratio": {"x": 0.5, "y": 1.5, "z": 0.4}
        }
    )

    return fig
#--------------------------------------------------------------

def drawMatrix(data_h, levels, calcF, data_f, track, leaps):
    x_max = len(levels)
    fig = go.Figure()
    
    # Heatmap для data_h
    fig.add_trace(go.Heatmap(
        z=data_h,
        xgap=1,
        ygap=1,
        hoverongaps=False,
        colorscale='Viridis',
        showscale=False,
        showlegend=True,
        name="Matrix-H"
    ))

    # Опциональный heatmap для data_f
    if calcF:
        fig.add_trace(go.Heatmap(
            z=data_f,
            xgap=1,
            ygap=1,
            hoverongaps=False,
            opacity=0.75,
            showscale=False,
            showlegend=True,
            name="Matrix-F"
        ))

    # Scatter для track
    fig.add_trace(go.Scatter(
        x=track[0],
        y=track[1],
        mode='lines+markers',
        marker=dict(size=8),
        showlegend=True,
        name="Track #"
    ))

    # Heatmap для leaps
    fig.add_trace(go.Heatmap(
        z=leaps,
        xgap=1,
        ygap=1,
        hoverongaps=False,
        opacity=0.85,
        colorscale='rainbow',
        showscale=False,
        showlegend=True,
        name="Leaps map"
    ))

    # Настройки осей
    fig.update_xaxes(
        range=[-1, x_max+1],
        showspikes=True,
        spikemode='across',
        showgrid=True,
        tick0=1, # Основные деления каждые 1 единицу
        gridwidth=1,
        gridcolor='LightGrey',
        minor_dtick=0.2, # Второстепенные деления каждые 0.2 единицы
        minor_gridwidth=0.5,
        minor_gridcolor='rgba(0,0,0,0.1)', # Более светлый цвет
        tickfont=dict(size=14)
    )
    fig.update_yaxes(
        autorange="reversed",
        title_text="Events deep",
        title_font=dict(size=16),
        showspikes=True,
        spikemode='across',
        showgrid=True,
        tick0=1, # Основные деления каждые 1 единицу
        gridwidth=1,
        gridcolor='LightGrey',
        minor_dtick=0.2, # Второстепенные деления каждые 0.2 единицы
        minor_gridwidth=0.5,
        minor_gridcolor='rgba(0,0,0,0.1)', # Более светлый цвет
        tickfont=dict(size=14)
    )

    fig.update_layout(
        template="plotly_dark",
        clickmode='event+select',
        margin=dict(t=20, l=20, r=20, b=20),
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            orientation="h"
        ),
        coloraxis_showscale=False,
        #height=400
    )

    return fig
#--------------------------------------------------------------------

def createCombinedSubplot(
    data_h, calcF, data_f, track, track_num, leaps, levels_fft,
    diff_waves, matrix_waves, track_zones, subplots_balance, deep, deep_mode
):
    x_max = len(diff_waves)
    d = levels_fft[deep]
    fft_y = d['fft_y']
    
    # Создаем сабплот с 2 строками, 1 колонкой
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=subplots_balance,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=["Events Depth Levels", "Tracks Matrix"]
    )

    # --- 1-я фигура: drawDeepLevelWaves ---
    deep_fig = drawDeepLevelWaves(diff_waves, matrix_waves, fft_y, track_zones, deep, deep_mode)
    for trace in deep_fig.data:
        fig.add_trace(trace, row=1, col=1)
    fig.update_layout(
        shapes=deep_fig.layout.shapes,
        annotations=deep_fig.layout.annotations
    )
    fig.update_xaxes(
        title_text="x",
        range=[-1, x_max+1],
        showgrid=True,
        showspikes=True,
        spikemode='across',
        autorange=False,
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Wave value",
        showgrid=True,
        showspikes=True,
        spikemode='across',
        row=1, col=1
    )

    # --- 2-я фигура: матрица через createMatrixFigure ---
    matrix_fig = drawMatrix(data_h, diff_waves, calcF, data_f, track, leaps)

    # Добавляем все треки из matrix_fig во 2 сабплот
    for trace in matrix_fig.data:
        fig.add_trace(trace, row=2, col=1)

    # Добавляем горизонтальную линию выбранной глубины из matrix_fig (в shapes и аннотациях она будет последней)
    # Поэтому добавим их вручную:
    fig.add_hline(
        y=deep,
        line_color='yellow',
        line_dash='dot',
        opacity=0.5,
        row=2, col=1,
        #annotation_text=f"Deep {deep}",
        #annotation_position="top right"
    )

    # Общие настройки
    fig.update_layout(
        template="plotly_dark",
        #clickmode='event+select',
        height=900,
        margin=dict(t=40, l=20, r=20, b=40),
        legend=dict(
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            orientation="h"
        ),
        coloraxis_showscale=False,
    )

    # Настройки осей
    fig.update_xaxes(
        range=[-1, x_max+1],
        showspikes=True,
        spikemode='across',
        showgrid=True,
        tickfont=dict(size=14),
        row=2, col=1,
        rangeslider=dict(visible=True, thickness=0.05)
    )
    fig.update_yaxes(
        autorange="reversed",
        title_text="Events deep",
        title_font=dict(size=16),
        showspikes=True,
        spikemode='across',
        showgrid=True,
        tickfont=dict(size=14),
        row=2, col=1
    )
    fig.update_yaxes(
        showspikes=True,
        spikemode='across',
        showgrid=True,
        row=1, col=1
    )
    fig.update_xaxes(
        showspikes=True,
        spikemode='across',
        row=1, col=1
    )

    return fig
#-----------------------------------------