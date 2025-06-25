import dash
from dash import Dash, html, dcc, callback, Input, Output, State, Patch, no_update
from plotly.subplots import make_subplots
import plotly.graph_objects as go


app = Dash(__name__)

# -----------------------------------------------------------------------------
# 1) Инициализируем наши данные:
# -----------------------------------------------------------------------------
# Пример levels_fft (win_num=0)
levels_fft = {
    0: [
        dict(fft_y=[0.4, -0.7], wave_y=[1.1, 2.2]),
        dict(fft_y=[1.4, -1.7], wave_y=[0.5, 1.5]),
        # ... сколько у вас уровней
    ]
}
win_num = 0

# Начальная фигура
fig = make_subplots(rows=2, cols=1, shared_xaxes=False)

# --- Сабплот 1: Heatmap, H-line, Scatter, Clicks map
z0 = [[1, 2], [3, 4]]             # просто пример
fig.add_trace(go.Heatmap(z=z0, name="Heatmap"), row=1, col=1)
fig.add_trace(go.Scatter(x=[0,1], y=[1,1], mode="lines", name="H-line"), row=1, col=1)
fig.add_trace(go.Scatter(x=[0,1], y=[1,2], mode="markers", name="Scatter1"), row=1, col=1)
# новый слой – тепловая карта кликов, изначально None
# dims той же, что и z0
z_clicks = [[None]*len(z0[0]) for _ in z0]
fig.add_trace(go.Heatmap(z=z_clicks, 
                         name="Clicks map", 
                         colorscale="Blues", showscale=False,
                         zmin=0, zmax=1), row=1, col=1)

# --- Сабплот 2: два бара и Wave
fig.add_trace(go.Bar(x=[0,1], y=[0.5, 1.2], name="FFT > 0"), row=2, col=1)
fig.add_trace(go.Bar(x=[0,1], y=[-0.3, -0.8], name="FFT < 0"), row=2, col=1)
fig.add_trace(go.Scatter(x=[0,1], y=[2,3], mode="markers", name="Wave"), row=2, col=1)

# -----------------------------------------------------------------------------
# 2) Layout с двумя Store
# -----------------------------------------------------------------------------
app.layout = html.Div([
    # храним levels_fft
    dcc.Store(id="store-levels", data=levels_fft),
    # храним clicks_map + last_y
    dcc.Store(id="store-clicks", data={
        "map": z_clicks,
        "last_y": None
    }),

    dcc.Graph(id="id-matrix", figure=fig)
])

# -----------------------------------------------------------------------------
# 3) Один колбэк на любой клик по Graph
# -----------------------------------------------------------------------------
@callback(
    Output("id-matrix", "figure", allow_duplicate=True),
    Output("store-clicks", "data"),
    Input("id-matrix", "clickData"),
    State("id-matrix", "figure"),
    State("store-levels", "data"),
    State("store-clicks", "data"),
    prevent_initial_call=True
)
def on_click(clickData, fig, store_levels, store_clicks):
    pt = clickData["points"][0]
    curve = pt["curveNumber"]
    x_click = int(pt["x"])
    y_click = pt["y"]
    yaxis_id = pt["yaxis"]  # обычно "y" для row=1, "y2" для row=2

    patches = []
    new_store = store_clicks.copy()

    # --- СПЛОТ 1: перенесли уровень y, пересчитали FFT и Wave
    if yaxis_id == "y":
        lvl = int(y_click)
        # 1) H-line на новый уровень
        idx_hline = next(i for i,d in enumerate(fig["data"]) if d["name"]=="H-line")
        p_h = Patch()
        p_h["data"][idx_hline]["y"] = [lvl, lvl]
        patches.append(p_h)

        # 2) FFT и Wave из store_levels
        data_l = store_levels[str(win_num) if isinstance(store_levels,dict) else win_num][lvl]
        fft_y  = data_l["fft_y"]
        wave_y = data_l["wave_y"]
        # split pos/neg
        pos = [v if v>0 else 0 for v in fft_y]
        neg = [v if v<0 else 0 for v in fft_y]

        idx_p = next(i for i,d in enumerate(fig["data"]) if d["name"]=="FFT > 0")
        idx_n = next(i for i,d in enumerate(fig["data"]) if d["name"]=="FFT < 0")
        idx_w = next(i for i,d in enumerate(fig["data"]) if d["name"]=="Wave")

        p2 = Patch(); p2["data"][idx_p]["y"] = pos; patches.append(p2)
        p3 = Patch(); p3["data"][idx_n]["y"] = neg; patches.append(p3)
        p4 = Patch(); p4["data"][idx_w]["y"] = wave_y; patches.append(p4)

        # запомним выбранный уровень
        new_store["last_y"] = lvl
        return patches, new_store

    # --- СПЛОТ 2: обновляем только Heatmap "Clicks map" и маркер Wave
    else:
        # забираем ранее выбранный lvl из сабплота1
        lvl = new_store.get("last_y")
        if lvl is None:
            # если не было клика в первом сабплоте – ничего не делаем
            return no_update, no_update

        # Toggle: если было None -> ставим 1, иначе сбрасываем в None
        cmap = [row[:] for row in new_store["map"]]
        cmap[lvl][x_click] = 1 if cmap[lvl][x_click] is None else None
        new_store["map"] = cmap

        # 1) обновляем Heatmap "Clicks map"
        idx_c = next(i for i,d in enumerate(fig["data"]) if d["name"]=="Clicks map")
        p_c = Patch()
        p_c["data"][idx_c]["z"] = cmap
        patches.append(p_c)

        # 2) выделяем точку в Wave
        idx_w = next(i for i,d in enumerate(fig["data"]) if d["name"]=="Wave")
        # строим списки размеров и цветов
        base_x = fig["data"][idx_w]["x"]
        L = len(base_x)
        sizes = [15 if i==x_click else 8 for i in range(L)]
        colors= ["red" if i==x_click else "blue" for i in range(L)]
        p_w2 = Patch()
        p_w2["data"][idx_w]["marker.size"]  = sizes
        p_w2["data"][idx_w]["marker.color"] = colors
        patches.append(p_w2)

        return patches, new_store

if __name__ == "__main__":
    app.run(debug=True)