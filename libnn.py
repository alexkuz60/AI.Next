import time
#import pprint
import numpy as np
import pandas as pd
#import os.path

from autogluon.tabular import TabularDataset, TabularPredictor

def autoSelectNextTrack(csv_data, hist_back, order):
    track_num = 0
    if hist_back > 0:
        track_num = csv_data[hist_back-1, order] - 1
    return track_num
#--------------------------------------------------------------

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
    Возвращает список всех треков.
    all_tracks[i] — результат для track_num=i+1
    """
    n_tracks = cumulative_data.shape[1]
    all_tracks = []
    leap_points = []
    for i in range(n_tracks):
        arr_t, arr_l = extract_track_xy(separate_data, cumulative_data, i)
        all_tracks.append(arr_t)
        leap_points.append(arr_l)
    
    return all_tracks, leap_points
#---------------------------------------------------------

def calcAllTracks(order_data, matrix_size, max_nums):
    # Создаем one-hot матрицу за одну операцию
    mark_tracks = np.zeros((matrix_size, max_nums), dtype=np.int32)
    
    # Используем меньший тип данных
    np.put_along_axis(mark_tracks, (order_data - 1)[:, None], 1, axis=1)

    # Накопительная сумма с явным указанием выходного типа
    sum_tracks = np.cumsum(mark_tracks, axis=0, dtype=np.int32)

    return mark_tracks, sum_tracks
#--------------------------------------------------------------

def calcPassport(matrix_data, max_deep):
    # Векторная реализация с flat-индексами
    rows = np.repeat(np.arange(matrix_data.shape[0]), matrix_data.shape[1])
    cols = matrix_data.ravel()
    passport = np.zeros((matrix_data.shape[0], max_deep), dtype=int)
    np.add.at(passport, (rows, cols), 1)

    return passport
#----------------------------------------------------------

def calcLeapPoints(leap_points, max_track):
    leaps_zones = []
    for tr in range(max_track):
        #tr_counter = 0
        points = leap_points[tr]
        points_x = points[0, :]
        tr_zones = points_x[1:-1]
        leaps_zones.append(tr_zones)
        #print(f"track{tr} zones {tr_zones}")

    return leaps_zones
#---------------------------------------

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

def matchRelationMatrix(c_data, r_data):
    """
    c_data = [cp, cc, cn]
    r_data = [rp, rc, rn]
    """
    mr=np.full(shape=(3,3), fill_value=0, dtype=np.int16)    #fill_value=np.nan

    #Значения циан-графика [х-1,х,х+1]
    mr[0,:] = c_data
    mr[1,:] = r_data
    # diff_cr:
    mr[2,:] = c_data - r_data

    diff_c = mr[0,:]
    diff_r = mr[1,:]
    
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
    #pp=pprint.PrettyPrinter(width=200)
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

def matchPrognosis(diff_waves, matrix_levels, leaps_zones):
    #print(leaps_zones)
    #Выборка из волновых функций значений в зонах трека:
    #Пустой словарь-массив матриц отношений:
    matrix_dict = []
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
        mr_track_map = []
        x_prev = 0
        for x in track_zones:
            rw = diff_waves[:, deep] # red line
            cw = matrix_levels[:, deep] # cyan line
            odd = deep%2
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
                mr = matchRelationMatrix(np.array([cp, cc, cn]), np.array([rp, rc, rn]))
                #row = {'tr': tr+1, 'x': x, 'x_freq': x_freq, 'y': deep, 'max_deep': max_deep, 'mr': mr, 'odd': odd}
                row = [tr+1, x, x_freq, deep, max_deep, cp, cc, cn, rp, rc, rn, odd]
                #row = [tr+1, x, x_freq, deep, max_deep, mr, odd]
                matrix_dict.append(row)

                new_mr = np.array(mr)
                mr_name = add_matrix_if_not_exists(mr_list, new_mr)
                mr_track_map.append(mr_name)
            deep += 1

        mr_map.append(mr_track_map)
        #карта матриц для треков построчный вывод
        #print(f"track {tr}: {mr_track_map}")

    #print('chances', chances)
    #print("mr_count:", len(mr_list))
    #print('matrix_dict', matrix_dict)
    #mr_table = make_markers_map_with_indices(mr_map)
    
    return mr_map, matrix_dict
#----------------------------------------------------------


#==========Сбор данных и формирование датасета (и словаря характеристик???)===========

def makeDataAndTestSets(file_name = "4x20.csv", matrix_size = 150, max_hback = 200, sorted = True):
    start = time.time()

    #TODO: Проверка на самый длинный ноль в архиве, иначе ошибка
    
    # Загрузка файла в массив NumPy
    with open(file_name, 'r', encoding='utf-8-sig') as f:
        csv_data = np.loadtxt(f, delimiter='\t').astype(int)

    #Характеристики архива:
    max_size = csv_data.shape[0] - max_hback
    max_orders = csv_data.shape[1]
    max_nums = csv_data.max()

    if matrix_size > max_size:
        matrix_size = max_size

    # Сортировка по возрастанию:
    if sorted:
        csv_data = np.sort(csv_data, axis=1)

    # Цикл создания матриц для всех ордеров в каждом h_back < max_hback:
    train_dataset = []
    test_dataset = []
    
    for hback in range(1, max_hback):
        matrix_data = np.array(csv_data[hback : matrix_size + hback, :], dtype=int)
        for order in range(max_orders):#
            next_num = autoSelectNextTrack(csv_data, hback, order)
            #print('next:', next_num+1)
            order_data = matrix_data[:, order]

            #Подготовка данных всех треков (по заданным габаритам матрицы) из данных файла для выбранного ордера
            mark_tracks, sum_tracks = calcAllTracks(order_data, matrix_size, max_nums)
            # Определяем максимальную глубину матрицы из таблицы треков:
            max_deep = np.max(sum_tracks) + 1

            #--------Рассчет паспорта матрицы:
            passport_h = calcPassport(sum_tracks, max_deep)

            #Рассчет точек отрисовки линий на матрице для всех треков и 2D-массив координат на матрице точек перехода
            _, leap_points = extract_all_tracks(mark_tracks, sum_tracks)
            #Точки глубинных переходов для всех треков
            leaps_zones = calcLeapPoints(leap_points, max_nums)
            #print("leap_zones", leaps_zones)

            # Создаем Deep waves для всех треков
            matrix_levels, diff_waves = calcWavesForDeepLevels(passport_h, max_deep)#, track)

            #Оценка шансов всех треков на статус трек-лидера
            mr_map, matrix_dict = matchPrognosis(diff_waves, matrix_levels, leaps_zones)
            #row = {'hback': hback, 'ord': order, 'next': next_num, 'matrix': matrix_dict}
            row = [hback, order, next_num+1, matrix_dict]
            if hback == 1:
                test_dataset.append(row)
            else:
                train_dataset.append(row)
    csv_header=['hback', 'order', 'next', 'matrix']
    train_df = pd.DataFrame.from_dict(train_dataset)
    train_df.to_csv('train_data.csv', header=csv_header)
    test_df = pd.DataFrame.from_dict(test_dataset)
    test_df.to_csv('test_data.csv', header=csv_header)

    calc_time = time.time() - start
    print ('Prepare DataSet:', calc_time)
#-----------------------------------------------------------------------

def trainModels(hist_data = 'train_data.csv', label = 'next'):
    #Training:
    start = time.time()

    train_data = TabularDataset(hist_data)
    train_data.head()
    train_data[label].describe()

    predictor = TabularPredictor(label=label).fit(
        train_data,
        num_cpus=32, num_gpus=1,
        presets='best',
        verbosity = 2, time_limit=60
    )
    
    calc_time = time.time() - start
    print (calc_time)

    return predictor
#--------------------------------------------------------------

def predictNewData(predictor, model_path = "./AutogluonModels/ag-20250611_194650", data_path = "test_data.csv", label = 'next'):   # 
    #Prognosis:
    #predictor = TabularPredictor.load(model_path)
    predictor=predictor
    test_data = TabularDataset(data_path)
    y_pred = predictor.predict(test_data.drop(columns=[label]))
    y_pred.head()
    print('predict:\n', y_pred)
#---------------------------------------------------------------
makeDataAndTestSets()
predictor = trainModels()
predictNewData(predictor)