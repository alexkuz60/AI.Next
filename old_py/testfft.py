import time
import json
import numpy as np
from scipy.signal.windows import gaussian
import plotly.graph_objects as go

def get_window(N, window_type='rectangular'):
    if window_type == 'hamming':
        return np.hamming(N)
    elif window_type == 'hann':
        return np.hanning(N)
    elif window_type == 'gaussian':
        return gaussian(N, std=0.4*N)
    else:
        return np.ones(N)

def prepare_wave_fft_data(tracks, window_type='rectangular'):
    """
    tracks: список из NumPy-массивов формы (N, M)
    Возвращает словарь data[track_idx][chan_idx] = {'x', 'wave_y', 'fft_y'}
    """
    t1 = time.time()
    data = {}
    N, M = tracks[0].shape
    x = np.arange(N)

    for t_idx, track in enumerate(tracks):
        # 1) ресэмплинг N->2N->N для wave_y
        wave2   = np.repeat(track, 2, axis=0)  # (2N, M)
        wave_ds = wave2[::2, :]               # (N, M)
        masks   = np.sign(wave_ds)            # +1/0/−1

        # 2) FFT на исходном сигнале + окно
        window    = get_window(N, window_type)
        win_track = track * window[:, None]
        fft_full  = np.fft.fft(win_track, axis=0)
        abs_fft   = np.abs(fft_full)
        log_fft   = np.log2(abs_fft + 1e-12)

        data[t_idx] = {}
        for c in range(M):
            wave_y = wave_ds[:, c]

            fft_y = log_fft[:, c].copy()
            fft_y -= np.nanmin(fft_y)         # минимум → 0
            fft_y *= masks[:, c]             # применяем маску
            fft_y = fft_y.astype('float')    # чтобы можно было вставить None
            fft_y[fft_y == 0] = None         # нули → None

            # масштабируем fft_y в тот же Y-диапазон, что и wave_y
            valid = ~np.isnan(fft_y)
            if np.any(valid):
                fmin, fmax = np.nanmin(fft_y), np.nanmax(fft_y)
                wmin, wmax = np.min(wave_y), np.max(wave_y)
                denom = (fmax - fmin) or 1.0
                fft_y[valid] = (fft_y[valid] - fmin)/denom
                fft_y[valid] = fft_y[valid] * (wmax - wmin) + wmin

            data[t_idx][c] = {'x': x, 'wave_y': wave_y, 'fft_y': fft_y}
    print(time.time()-t1)
    return data

# === Основной блок ===

if __name__ == '__main__':
    # 1) Загружаем ваш файл levels.json
    with open('levels.json', 'r', encoding='utf-8-sig') as f:
        arr = np.array(json.load(f))   # shape (N, 13)

    tracks = [arr]      # один трек из 13 колонок
    data   = prepare_wave_fft_data(tracks, window_type='rectangular')

    # Для примера возьмём канал 0
    d       = data[0][5]
    x       = d['x']
    wave_y  = d['wave_y']
    fft_y   = d['fft_y']

    # Разделяем positive/negative для цветных баров
    pos_mask = (fft_y is not None) & (fft_y > 0)
    neg_mask = (fft_y is not None) & (fft_y < 0)

    # В Plotly None пропускаются автоматически
    # Собираем фигуру
    fig = go.Figure()

    # Положительные FFT-бары
    fig.add_trace(go.Bar(
        x=x[pos_mask],
        y=fft_y[pos_mask],
        marker_color='rgba(0,128,255,0.5)',
        name='FFT > 0',
    ))
    # Отрицательные FFT-бары
    fig.add_trace(go.Bar(
        x=x[neg_mask],
        y=fft_y[neg_mask],
        marker_color='rgba(255,0,0,0.5)',
        name='FFT < 0',
    ))

    # Сплайн-линия для wave_y
    fig.add_trace(go.Scatter(
        x=x,
        y=wave_y,
        mode='lines',
        line=dict(shape='spline', width=2, color='black'),
        name='Wave (spline)',
    ))

    # Оформление
    fig.update_layout(
        title='Wave vs FFT (channel 0)',
        xaxis_title='History',
        xaxis_spikemode='across',
        yaxis_spikemode='across',
        yaxis_title='Amplitude',
        barmode='overlay',      # бары поверх линии
        template='simple_white'
    )
    
    # Показываем
    fig.show()