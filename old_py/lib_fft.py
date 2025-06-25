import json
import numpy as np
import plotly.graph_objs as go
from scipy.signal.windows import gaussian

def save_tables_to_json(tables, filename):
    # Преобразуем каждый numpy массив в список списков
    tables_list = [table.tolist() for table in tables]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tables_list, f, ensure_ascii=False, indent=2)

def load_tables_from_json(filename):
    with open(filename, 'r', encoding='utf-8-sig') as f:
        tables_list = json.load(f)  # загружается список таблиц или одна таблица
    # Если в JSON одна таблица (список списков), обернем в список
    if isinstance(tables_list, list) and len(tables_list) > 0 and isinstance(tables_list[0], list):
        # Проверяем, является ли первый элемент таблицей (списком списков)
        if not isinstance(tables_list[0][0], list):
            # Одна таблица, оборачиваем в список
            tables_list = [tables_list]
    # Преобразуем в numpy массивы
    tables = [np.array(table) for table in tables_list]
    return tables

# Пример использования:
# save_tables_to_json(T, 'tables.json')
# T_loaded = load_tables_from_json('tables.json')


def get_x_fft(N):
    return np.arange(1, N, 2)

def get_window(N, window_type):
    if window_type == 'hamming':
        return np.hamming(N)
    elif window_type == 'hann':
        return np.hanning(N)
    elif window_type == 'gaussian':
        return gaussian(N, std=0.4*N)
    else:
        return np.ones(N)  # прямоугольное окно

def prepare_wave_fft_data(tables, window_type='rectangular'):
    data = {}
    #Дублирование значений wave for fft:
    #re_tables = np.array(tables)
    #N, M = re_tables[0].shape
    #print(tables[0].shape)
    #re_tables = np.repeat(tables, 2, axis=1)
    #print(re_tables[0].shape)
    
    N, M = tables[0].shape
    x_wave = np.arange(N)
    x_fft = np.arange(1, N, 2)
    for t_idx, table in enumerate(tables): #for t_idx, table in enumerate(re_tables):
        window = get_window(N, window_type)
        windowed_table = table[t_idx] * window[:, np.newaxis]
        fft_table = np.fft.fft(windowed_table, axis=0)
        data[t_idx] = {}
        for c_idx in range(M):
            wave_y = table[:, c_idx] #table
            fft_y_full = np.abs(fft_table[:, c_idx])
            fft_y = fft_y_full[1:len(x_fft)+1]
            data[t_idx][c_idx] = {
                'wave_y': wave_y,
                'fft_y': fft_y,
                'x_fft': x_fft,
                'x_wave': x_wave
            }
    return data

def create_wave_fft_plot(data_array, table_num, col_num):
    """
    Создаёт Plotly фигуру с двумя треками: волна (сплайн, толщина 2) и FFT (гистограмма),
    масштабируя FFT по максимуму волны.
    """
    wave_y = data_array[table_num][col_num]['wave_y']
    fft_y = data_array[table_num][col_num]['fft_y']
    x_fft = data_array[table_num][col_num]['x_fft']
    x_wave = data_array[table_num][col_num]['x_wave']

    # Создание маски
    #mask = np.sign(wave_y)

    max_wave = np.max(np.abs(wave_y))
    #wave_y = wave_y * mask
    fft_y = np.log2(np.abs(fft_y))
    max_fft = np.max(fft_y)
    zoom = max_wave / max_fft if max_fft != 0 else 1.0
    fft_y_scaled = fft_y * zoom

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=wave_y,
        x=x_wave,
        mode='lines',
        line=dict(shape='spline', width=2, color='yellow'),
        name=f'Волна (табл {table_num}, столбец {col_num})',
        legendgroup='wave',
        showlegend=True
    ))

    fig.add_trace(go.Bar(
        y=fft_y_scaled,
        x=x_fft,
        name='FFT (масштабировано)',
        marker_color='aliceblue',
        opacity=0.6,
        legendgroup='fft',
        showlegend=True
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f'Волна и FFT для таблицы {table_num}, столбца {col_num}',
        xaxis=dict(
            title='Индекс',
            showgrid=True,
            zeroline=True,
            zerolinecolor='LightPink',
            zerolinewidth=2,
            fixedrange=False
        ),
        yaxis=dict(
            title='Амплитуда',
            showgrid=True,
            fixedrange=False
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(0,0,0,0)'
        ),
        bargap=0.2,
        height=750,
        margin=dict(t=80)
    )

    return fig