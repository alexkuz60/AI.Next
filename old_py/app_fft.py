from dash import Dash, dcc, Input, Output
import dash_mantine_components as dmc
from lib_fft import prepare_wave_fft_data, create_wave_fft_plot
import lib_fft

# Загрузка таблиц (пример)
# tables = load_tables_from_json('levels.json')

T = lib_fft.load_tables_from_json('levels.json')
# Для примера:
#T = [np.random.randint(-20, 21, size=(200, 5)) for _ in range(3)]

app = Dash(__name__, title="FFT Viewer with Window Selection")

app.layout = dmc.MantineProvider(
    theme={"colorScheme": "dark"},
    defaultColorScheme="dark",
    children=[
        dmc.Container(
            size="xl",
            children=[
                dmc.Title("FFT Viewer", order=1, mb="md"),
                dmc.Group(
                    children=[
                        dmc.Select(
                            id="table-select",
                            label="Выберите таблицу",
                            data=[{"label": f"Таблица {i}", "value": str(i)} for i in range(len(T))],
                            value="0",
                            style={"minWidth": 150}
                        ),
                        dmc.Select(
                            id="column-select",
                            label="Выберите столбец",
                            data=[{"label": f"Столбец {i}", "value": str(i)} for i in range(T[0].shape[1])],
                            value="0",
                            style={"minWidth": 150}
                        ),
                        dmc.Select(
                            id="window-select",
                            label="Выберите окно",
                            data=[
                                {"label": "Прямоугольное (без окна)", "value": "rectangular"},
                                {"label": "Хэмминга", "value": "hamming"},
                                {"label": "Ханна", "value": "hann"},
                                {"label": "Гаусса", "value": "gaussian"},
                            ],
                            value="rectangular",
                            style={"minWidth": 200}
                        ),
                    ],
                    mb="md"
                ),
                dcc.Graph(id="fft-graph", style={"width": "100%", "height": "500px"})
            ]
        )
    ]
)

@app.callback(
    Output("fft-graph", "figure"),
    Input("table-select", "value"),
    Input("column-select", "value"),
    Input("window-select", "value"),
)
def update_graph(table_num_str, col_num_str, window_type):
    table_num = int(table_num_str)
    col_num = int(col_num_str)
    data_array = prepare_wave_fft_data(T, window_type=window_type)
    return create_wave_fft_plot(data_array, table_num, col_num)

if __name__ == "__main__":
    app.run(debug=True)