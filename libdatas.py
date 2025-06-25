import time
import base64
import math
import pprint

import numpy as np
from scipy.signal.windows import gaussian
import pandas as pd

import libgraphs as lg

#=========================================================================

def readFileContent(content, sorted):
    content_type, content_string = str(content).split(',', 1)
    decoded = base64.b64decode(content_string)
    csv_data = decoded.decode('utf-8-sig').splitlines()
    arr_data = np.loadtxt(csv_data, delimiter='\t').astype(int)
    matrix_max = arr_data.shape[0]
    max_order = arr_data.shape[1]

    # Сортировка по возрастанию:
    if sorted:
        arr_data = np.sort(arr_data, axis=1)
    
    return arr_data, matrix_max, max_order
#--------------------------------------------------------------

def setTrackList(track_num, max_nums, hist_back):
    track_list = [{"label": f"Track {i+1}", "value": str(i+1)} for i in range(max_nums)]
    #if hist_back > 0:
        #track_list = [{"label": f"{'Win ' if i == track_num-1 else ''}Track {i + 1}", "value": str(i+1)} for i in range(max_nums)]
        
    return track_list
#--------------------------------------------------------------

def reshapeMatrix(csv_data, matrix_data, new_size, hist_back):
    np.copyto(matrix_data, csv_data[hist_back : new_size + hist_back])
    return matrix_data
#--------------------------------------------------------------
def autoSelectNextTrack(csv_data, hist_back, order):
    track_num = 0
    if hist_back > 0:
        track_num = csv_data[hist_back-1, order] - 1
    return track_num

def uploadFileByName(file_name, order, matrix_size, hist_back, sorted=False):
    """
    CSV файл по умолчанию, закружаемый при входе на сайт/обновлении,
    чтобы не отрабатывать графики при отсутствии данных.
    Варианты: демо файл, рандомайзер-генератор, последний файл прошлой сессии...
    Пока - демо файл.
    """
    # Статовая загрузка  в массив NumPy
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        csv_data = np.loadtxt(f, delimiter='\t').astype(int)
    
    max_size = csv_data.shape[0] - hist_back
    max_order = csv_data.shape[1]
    max_nums = csv_data.max()
    
    if matrix_size > max_size:
        matrix_size = max_size

    # Сортировка по возрастанию:
    if sorted:
        csv_data = np.sort(csv_data, axis=1)

    #order_data = csv_data[hist_back : matrix_size + hist_back, order]
    matrix_data = np.array(csv_data[hist_back : matrix_size + hist_back, :], dtype=int)
    #np.copyto(matrix_data, csv_data[hist_back : matrix_size + hist_back])

    track_num = autoSelectNextTrack(csv_data, hist_back, order)

    #print('matrix_size::', matrix_size)
    return csv_data, matrix_data, matrix_size, max_size, max_order, max_nums, track_num
#--------------------------------------------------------------

def get_fft_window(N, window_type):
    if window_type == 'hamming':
        return np.hamming(N)
    elif window_type == 'hann':
        return np.hanning(N)
    elif window_type == 'gaussian':
        return gaussian(N, std=0.4*N)
    else:
        return np.ones(N)  # прямоугольное окно
#---------------------------------------------------------

def prepare_wave_fft_data(levels, window_type='rectangular'):
    max_x, max_deep = levels.shape
    
    #axis_x = np.arange(max_x).astype('float')
    #for t_idx, track in enumerate(track):
    #Ресэмплируем волновые данные всех глубин.уровней удвоенной частотой:
    wave2   = np.repeat(levels, 2, axis=0)
    #Лишнее??? можно взять levels 
    wave_ds = wave2[::2, :]
    #Маска положительных/отрицательных полуволн для всех глубин.уровней.
    masks   = np.sign(wave_ds)
    wave_ds_float = wave_ds.astype('float')

    #ФФТ-окна для всех глубин
    window = get_fft_window(max_x, window_type)
    win_track = levels * window[:, None]
    fft_full  = np.fft.fft(win_track, axis=0)
    abs_fft   = np.abs(fft_full)
    bias = 1e-12
    log_fft = np.log2(abs_fft + bias)

    data = {}

    fft_y_for_transitions = np.empty((max_x, max_deep), dtype='float')

    for c in range(max_deep):
        wave_y = wave_ds_float[:, c]

        fft_y = log_fft[:, c].copy()
        fft_y -= np.min(fft_y)
        fft_y_masked = fft_y * masks[:, c]
        fft_y_masked = fft_y_masked.astype('float')

        fft_y_for_transitions[:, c] = fft_y_masked

        fmin, fmax = np.min(fft_y_masked), np.max(fft_y_masked)
        wmin, wmax = np.min(wave_y), np.max(wave_y)
        denom = (fmax - fmin) or 1.0
        fft_y_masked = (fft_y_masked - fmin)/denom
        fft_y_masked = fft_y_masked * (wmax - wmin) + wmin

        pos_mask = (fft_y_masked >= 0)
        neg_mask = (fft_y_masked < 0)
        fft_y_2d = np.vstack((fft_y_masked * pos_mask, fft_y_masked * neg_mask))

        data[c] = {'fft_y': fft_y_2d}#'x': axis_x, 
    return data
#--------------------------------------------------------

def calcAllTracks(order_data, matrix_size, max_nums):
    # Создаем one-hot матрицу за одну операцию
    mark_tracks = np.zeros((matrix_size, max_nums), dtype=np.int32)
    print('m_size', matrix_size)
    # Используем меньший тип данных
    np.put_along_axis(mark_tracks, (order_data - 1)[:, None], 1, axis=1)

    # Накопительная сумма с явным указанием выходного типа
    sum_tracks = np.cumsum(mark_tracks, axis=0, dtype=np.int32)

    return mark_tracks, sum_tracks
#--------------------------------------------------------------

def calcIntentVector(X, Y, W, H):
    # Суммируем координаты
    #X = sum(p[0] for p in points)
    #Y = sum(p[1] for p in points)
    #points = np.array([X, Y])  # Преобразуем в numpy-массив
    
    # Длина суммарного вектора
    L = math.sqrt(X**2 + Y**2)
    if L == 0:
        return (0, 0)

    # Вычисляем максимально возможный масштаб по X и Y, чтобы не выйти за границы
    scale_x = W / X if X != 0 else float('inf')
    scale_y = H / Y if Y != 0 else float('inf')

    # Выбираем минимальный положительный масштаб
    scales = [s for s in [scale_x, scale_y] if s > 0]
    max_scale = min(scales) if scales else 1

    # Если длина вектора превышает максимально допустимую длину, масштабируем
    if max_scale < 1:
        X *= max_scale
        Y *= max_scale

    # Ограничиваем координаты в пределах прямоугольника
    X = max(0, min(X, W))
    Y = max(0, min(Y, H))-1

    return (X, Y)
#-----------------------------------------------------------------------------

def calcLeapPoints(passport, leap_points):
    # Подготовка матрицы переходов для 2-го субграфика матрицы & считаем 1 в треках.
    heatmap_leaps = np.zeros_like(passport)
    leap_rows, leap_cols = heatmap_leaps.shape
    max_track = len(leap_points)
    #Координаторы вектора намерений
    #vector_x = 0
    #vector_y = 0
    leaps_zones = []
    for tr in range(max_track):
        #tr_counter = 0
        points = leap_points[tr]
        points_x = points[0, :]
        points_y = points[1, :]
        tr_zones = points_x[1:-1]
        leaps_zones.append(tr_zones)
        #print(f"track{tr} zones {tr_zones}")
        
        #Получем координаты, исключив крайние точки линии трека:
        size_x = len(points_x) - 1
        for px in range(1, size_x):
            x = points_x[px]
            deep = points_y[px]

            # Занесение точки на шаблон матрицы + z-метка номера трека:
            heatmap_leaps[x, deep] = tr + 1

    """
            #Сборка вектора намерений:
            vector_x += x/(size_x)
            vector_y += deep/(size_x)
            tr_counter += 1
            #print(f'{x}:{deep}')

            # Коррекция суммарного вектора (ввод последних значений для треков)
            if px == len(points_x) - 2:
                last_x = px + 1
                vector_x += points_x[last_x]
                vector_y += points_y[last_x]

    print(f'X:{vector_x}  Y:{vector_y}')
    
    #intent_len = math.sqrt(vector_x**2 + vector_y**2)
    #Координаты вектора намерений:
    #XY = calcIntentVector(vector_x, vector_y, leap_rows-1, leap_cols-1)
    #print(f"VectorXY:{XY[0]} : {XY[1]}")
    """
    #Замена нулей на None и разворот для графика матрицы:
    heatmap_leaps = np.where(heatmap_leaps==0, np.nan, heatmap_leaps)
    heatmap_leaps = np.rot90(heatmap_leaps, k=1)
    heatmap_leaps = heatmap_leaps[::-1]

    return heatmap_leaps, leaps_zones
#---------------------------------------

def extract_track_xy(separate_data, cumulative_data, track_num):
    """
    Возвращает для одного трека (track_num, 1-based) массив shape=(2, n+2),
    где первая строка — X‑координаты, вторая — Y‑координаты.
    """
    tn = track_num - 1
    idx = np.flatnonzero(separate_data[:, tn])
    n = idx.size
    #print('idx',idx)
    x = np.concatenate(([0], idx-1, [cumulative_data.shape[0] - 1]))#prev_deep
    x = np.concatenate(([0], idx, [cumulative_data.shape[0] - 1]))
    y = np.zeros(2, dtype=int)  # только [0, 0], если нет точек
    if n > 0:
        y_prev = cumulative_data[idx-1, tn]
        y_vals = cumulative_data[idx, tn]
        y = np.concatenate(([0], y_prev, [y_prev[-1]]))#prev_deep
        y = np.concatenate(([0], y_vals, [y_vals[-1]]))
    #else:
    #    y = np.zeros(2, dtype=int)  # только [0, 0], если нет точек
    #    x = np.array([0, cumulative_data.shape[0] - 1], dtype=int)

    leap_points = np.vstack((x, y))
    #print('leap points', leap_points)

    # Создание нового массива с дополнительными значениями
    new_x = []
    new_y = []

    for i in range(len(x) - 1):
        if i > 0:  # Добавляем [x-1, y-1] перед каждой новой парой [x, y], ????кроме первой
            new_x.append(x[i] - 1)
            new_y.append(y[i] - 1)
            new_x.append(x[i])
            new_y.append(y[i])
        else:
            new_x.append(x[i])
            new_y.append(y[i])
    
    # Последняя точка (для дорисовки последней линии трека линии до конца графику
    new_x.append(len(separate_data)-1)
    new_y.append(y[-1])

    return np.vstack((new_x, new_y)), leap_points
#-----------------------------------------------------------

def extract_all_tracks(separate_data, cumulative_data):
    """
    Возвращает список и словарь всех треков.
    all_tracks[i] — результат для track_num=i+1
    all_tracks_dict[track_num] — результат для track_num
    """
    n_tracks = cumulative_data.shape[1]
    all_tracks = []
    leap_points = []
    for i in range(n_tracks):
        arr_t, arr_l = extract_track_xy(separate_data, cumulative_data, i)
        all_tracks.append(arr_t)
        leap_points.append(arr_l)
    
    #all_tracks = [extract_track_xy(separate_data, cumulative_data, i) for i in range(n_tracks)] #(separate_data, cumulative_data, i), leap_points

    #all_tracks_dict = {i+1: arr for i, arr in enumerate(all_tracks_list)}
    #print('all_tracks', all_tracks)
    return all_tracks, leap_points
#---------------------------------------------------------

def calcPassport(matrix_data, max_deep):
    # Векторная реализация с flat-индексами
    rows = np.repeat(np.arange(matrix_data.shape[0]), matrix_data.shape[1])
    cols = matrix_data.ravel()
    passport = np.zeros((matrix_data.shape[0], max_deep), dtype=int)
    np.add.at(passport, (rows, cols), 1)
    #print('passport', passport.shape)
    #print(passport)

    return passport
#----------------------------------------------------------

def matchDiffDeep(deep, deep_mode):
    diff_deep = 0
    if deep_mode == 1:
        diff_deep = deep
    elif deep_mode == 1 and deep > 0:
        diff_deep = deep - 1
    return diff_deep
#---------------------------------------------

def calcWavesMinMax(w1, w2, deep, deep_mode):
    diff_deep = matchDiffDeep(deep, deep_mode)
    # Оптимизируем вычисление y_min и y_max для корректного диапазона
    y_min = min(w1[:, deep].min(), w2[:, diff_deep].min())# - 1
    y_max = max(w1[:, deep].max(), w2[:, diff_deep].max())# + 1
    #print(f"min:{y_min} max:{y_max}")

    return y_min, y_max
#---------------------------------------------

def createShapesAndAnnotations(track_zones, y_min, y_max):
    # Используем batch добавление фигур и аннотаций для производительности
    shapes = []
    annotations = []
    number = 1
    for x in track_zones:
        #Зона поиска признаков трека-лидера:
        start = x-1
        end = x+1
        shapes.append(
            dict(type="rect", x0=start, x1=end, y0=y_min, y1=y_max, fillcolor="rgba(255,165,0,0.1)", line=dict(width=0.1))
        )
        #Линия точки перехода для выбранного трека
        shapes.append(
            dict(type="line", x0=x, x1=x, y0=y_min, y1=y_max, line=dict(color="orange", width=1, dash="dot"))
        )
        annotations.append(
            dict(x=x, y=y_max, text=f"#{number}:x{x}", showarrow=True, arrowhead=1, ax=0, ay=-40, bordercolor="orange", borderwidth=1, bgcolor="rgba(30,30,30,0.5)", font=dict(color="white", size=14))
        )
        number += 1

    return shapes, annotations
#----------------------------------------------

def calcWavesForDeepLevels(passport, max_deep): #, win_track):
    # next_val: для каждой строки, это passport, сдвинутый на 1 влево, а последний столбец — 0
    next_val = np.zeros_like(passport)
    next_val[:, :-1] = passport[:, 1:]  # для всех, кроме последнего столбца

    # event: 1 если n_track совпадает с глубиной, иначе 0
    #depth_indices = np.arange(max_deep)
    #track_events = (win_track[:, None] == depth_indices).astype(int)

    levels = passport - next_val# - track_events
    diff_waves = passport - levels# - 1

    return levels, diff_waves
#----------------------------------------------------------

def matchRelationMatrix(tr, deep, x, x_freq, c_data, r_data):
    """
    c_data = [cp, cc, cn]
    r_data = [rp, rc, rn]
    len_x = x - x_prev
    """
    mr=np.full(shape=(2,3), fill_value=0, dtype=np.int16)    #fill_value=np.nan

    #Значения циан-графика [х-1,х,х+1]
    mr[0,:] = c_data
    mr[1,:] = r_data
    diff_c = mr[0,:]
    diff_r = mr[1,:]
    
    diff_cr = c_data - r_data
    #Сокращаем кол-во типов матриц отношений МО в словаре МО..
    #..большое расхождение графиков считаем матрицей одного типа
    #diffs for prev, curr, next:
    diff_c[1] = mr[0,1] - mr[0,0]
    diff_c[2] = mr[0,2] - mr[0,0]
    if diff_c[0] > 2:
        diff_c[0] = 3
    elif diff_c[0] < -2:
        diff_c[0] = -3

    diff_r[1] = mr[1,1] - mr[1,0]
    diff_r[2] = mr[1,2] - mr[1,0]
    if diff_r[0] > 2:
        diff_r[0] = 3
    elif diff_r[0] < -2:
        diff_r[0] = -3

    #INFO: Results for win tracks
    pp=pprint.PrettyPrinter(width=200)
    #if tr==11 or tr==1 or tr==4 or tr==16:
        #pp.pprint(f"tr: {tr+1} deep: {deep} x: {x} >> cw: {mr[0]}  rw: {mr[1]}  diff_cr: {diff_cr}  diff_c: {diff_c} diff_r: {diff_r} odd: {deep%2}")
    #pp.pprint(f"{tr+1}, {deep}, {x}, {x_freq}, {mr[0]}, {mr[1]}, {diff_cr}, {diff_c}, {diff_r}, {deep%2}")
        #print(mr)
    return mr
#----------------------------------------------------------

def add_matrix_if_not_exists(mr_list, new_mr):
    for idx, mat in enumerate(mr_list):
        if np.array_equiv(mat, new_mr):
            return f"M{idx}"
    mr_list.append(new_mr)
    return f"M{len(mr_list)-1}"
#-----------------------------------------------------------
def format_markers_map_to_table(markers_map, max_track=19):
    # Получаем все маркеры (столбцы), сортируем по номеру после 'M'
    markers = sorted(markers_map.keys(), key=lambda x: int(x[1:]))
    #print(markers)
    # Создаем пустую таблицу с индексами треков и колонками маркеров
    df = pd.DataFrame(index=range(max_track + 1), columns=markers)
    # Переименовываем индекс для удобства
    df.index.name = 'track'
    # Заполняем таблицу
    for marker, entries in markers_map.items():
        for entry in entries:
            track = entry['track']
            pos = entry['pos']
            df.at[track, marker] = pos

    # Заменяем пустые ячейки на пустые списки или пустую строку
    df = df.fillna('')

    return df
#--------------------------------------------------------------
def make_markers_map_with_indices(mr_map):
    """
    Для каждого маркера определяет:
    - в каких подсписках он встречается (индексы подсписков)
    - сколько раз встречается в каждом подсписке
    - позиции (индексы) маркера внутри каждого подсписка

    :param list_of_lists: список списков с маркерами (строками)
    :return: словарь вида
        {
            'М1': [
                {'sublist_index': 0, 'count': 2, 'positions': [0, 3]},
                {'sublist_index': 2, 'count': 1, 'positions': [1]},
                ...
            ],
            'М2': [
                ...
            ],
            ...
        }
    """
    result = {}

    for mr_index, mr_list in enumerate(mr_map):
        # Для каждого маркера в подсписке считаем позиции
        positions_map = {}
        for pos, marker in enumerate(mr_list):
            positions_map.setdefault(marker, []).append(pos)

        # Обновляем общий результат
        for marker, positions in positions_map.items():
            if marker not in result:
                result[marker] = []
            result[marker].append({
                'track': mr_index,
                'count': len(positions),
                'pos': positions
            })
            #print(marker)
    #pprint.pprint(result)
    # Формируем таблицу
    mr_table = format_markers_map_to_table(result)
    return mr_table
#-----------------------------------------------------------

def chancesToPercentes(chances):
    arr_chances = np.array(chances)
    total = np.sum(arr_chances)
    prognosis = np.zeros_like(arr_chances)
    if total > 0:
        prognosis = 100 * arr_chances / total
        prognosis = np.around(prognosis, decimals=1)
    return prognosis
#--------------------------------------------------------------

def matchPrognosis(diff_waves, matrix_levels, leaps_zones):
    #Выборка из волновых функций значений в зонах трека:
    chances = []
    #Пустой словарь-массив матриц отношений:
    mr_list = []
    #Пустая карта найденных матриц отношений для всех треков:
    mr_map = []

    tracks_num = len(leaps_zones)
    #rw = diff_waves[:, 0] # red line STAZIS
    for tr in range(tracks_num):
        track_zones = leaps_zones[tr]
        #print(f"tr:{tr} points:{track_zones}")
        max_deep = len(track_zones)
        deep = 1
        sign_count = 0
        signs_weight = 0
        mr_track_map = []
        x_prev = 0
        for x in track_zones:
            rw = diff_waves[:, deep] # red line
            cw = matrix_levels[:, deep] # cyan line
            #odd = deep%2
            #Частотность(редкие события,нормально распределенные, частые, повторяющиеся=1):
            x_freq = x - x_prev
            x_prev = x
            #Анализируемые точки:
            if x < (len(rw)-1):
                #red--------
                rp = rw[x-1]
                rc = rw[x]
                rn = rw[x+1]
                #cyan-------
                cp = cw[x-1]
                cc = cw[x]
                cn = cw[x+1]
            else:
                rn = rw[x]
                cn = cw[x]

            #Строка матрицы oтношений:
            if (not(cp == cc ==cn)) and (not(rp==rc==rn)):
                mr = matchRelationMatrix(tr, deep, x, x_freq, np.array([cp, cc, cn]), np.array([rp, rc, rn]))
                new_mr = np.array(mr)
                mr_name = add_matrix_if_not_exists(mr_list, new_mr)
                mr_track_map.append(mr_name)

            #Элементарные правила:

            deep += 1
            signs_weight += sign_count

        #Общая оценка шансов трека:
        track_chance = 0
        if max_deep > 0:
            track_chance = signs_weight/max_deep
        chances.append(track_chance)

        mr_map.append(mr_track_map)
        #карта матриц для треков построчный вывод
        #print(f"track {tr}: {mr_track_map}")

    #print('chances', chances)
    print("mr_count:", len(mr_list))
    mr_table = make_markers_map_with_indices(mr_map)
    
    return chances, mr_table
#----------------------------------------------------------

def calcInverseMatrix(input_arr, matrix_size, max_nums, max_deep):
    # Инверсная Матрица 
    order_data_f = input_arr[::-1]
    tracks_f, arr_matrix_f = calcAllTracks(input_arr=order_data_f, matrix_size=matrix_size, cols=max_nums)
    # -->>> Далее все прочие расчеты для треков инверсной матрицы...
    #...
    #...
    #...
    passport_f = calcPassport(arr_matrix_f, max_deep)
    #-----Подготовка матрицы для графика:
    #Замена нулей в матрице на None
    matrix_f = passport_f
    matrix_f = np.where(matrix_f==0, np.nan, matrix_f)
    # Разворот матрицы для графика матрицы:
    matrix_f = np.rot90(matrix_f, k=-1)
    #matrix_f = matrix_f[::-1]
    return matrix_f
#---------------------------------------------------------------

def calcAllDatas(order_data, matrix_size, max_nums, track_num, deep_mode, subplots_balance=[0.5,0.5], calc_f=False):
    start = time.time()
    #Подготовка данных всех треков (по заданным габаритам матрицы) из данных файла для выбранного ордера
    mark_tracks, sum_tracks = calcAllTracks(order_data, matrix_size, max_nums) 
    # Определяем максимальную глубину матрицы из таблицы треков:
    max_deep = np.max(sum_tracks) + 1
    
    #--------Рассчет паспорта матрицы:
    passport_h = calcPassport(sum_tracks, max_deep)
    
    #Рассчет точек отрисовки линий на матрице для всех треков и 2D-массив координат на матрице точек перехода
    tracks_points, leap_points = extract_all_tracks(mark_tracks, sum_tracks)
    #Точки глубинных переходов для всех треков
    heatmap_leaps, leaps_zones = calcLeapPoints(passport_h, leap_points)
    #print("leap_zones", leaps_zones)

    #отображаемый на графике выбранный трек (0 он же ("1"-й) по умолчанию на старте)
    track = tracks_points[track_num]

    #print(f'track n{track_num}:', tracks_points)
    
    # Создаем Deep waves для всех треков
    matrix_levels, diff_waves = calcWavesForDeepLevels(passport_h, max_deep)#, track)
    #print("diff_pass_shape", diff_waves.shape)

    # Создаем Deep fft для всех треков:
    levels_fft = prepare_wave_fft_data(matrix_levels)
    levels_store = {'diff_waves': diff_waves, 'matrix_levels': matrix_levels, 'levels_fft': levels_fft, 'leaps_zones': leaps_zones}
    
    #Оценка шансов всех треков на статус трек-лидера
    #chances = calcTracksChances(tracks_chances, tracks_points=tracks_points)
    chances, mr_table = matchPrognosis(diff_waves, matrix_levels, leaps_zones)
    # Выводим вероятностные результаты треков в гистограмму шансов
    chances_bar = lg.drawChancesBar(chances, max_nums, track_num)

    #---------Подготовка матрицы для графика:
    #Замена нулей в матрице на None
    matrix_h = passport_h
    matrix_h = np.where(passport_h==0, np.nan, passport_h)
    
    # Разворот матрицы для графика матрицы:
    matrix_h = np.rot90(matrix_h, k=1)
    matrix_h = matrix_h[::-1]

    #-----------------Инверсная матрица (из прошлого в будущее):
    if calc_f:
        matrix_f = calcInverseMatrix(order_data[::-1], matrix_size, max_nums, max_deep)
    else:
        matrix_f=[]
    
    track_zones = leaps_zones[0]
    matrix_graf = lg.createCombinedSubplot(
        matrix_h, calc_f, matrix_f,
        track, track_num,
        heatmap_leaps,
        levels_fft, diff_waves, matrix_levels,
        track_zones, #intersections,
        subplots_balance,
        deep=0,
        deep_mode=0
    )
    calc_time = time.time()-start
    print (calc_time)

    return matrix_graf, chances_bar, levels_store, mr_table, calc_time
#---------------------------------------------------------------
