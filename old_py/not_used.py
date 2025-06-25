def calcTracksChances(marked_points, tracks_points): 
    V = len(tracks_points)
    #print('V', V)
    #track_max_deep = max(tracks_points[1])
    #rows, cols = fined_track.shape
    match_percents = []
    
    for v in range(V):
        i_track = tracks_points[v][0]

        #Максимальная глубина уровня трека:
        deep_y = tracks_points[v][1]

        track_max_deep = max(deep_y)
        #print('max_y', track_max_deep)
        
        """
        Ищем совпадения маркеров-аннотаторов level_points в треке i_track с точками начала/кончала трека 
        для каждого уровня, с учётом что:
            1-я х-координата - нечетная в треке, 2-я четная.
            х[0] и x[-1] в i_track[0..-1] - точки замыкания для рисования трека
            todo: добавить вес  /= кол-во треков z, проходящих через точку
        """
        counter_x0 = 0
        counter_x1 = 0
        for deep_x in range(1, track_max_deep, 2):
            #Точки трека deep_x для индекса уровня глубины deep_y
            level_points = marked_points[deep_y[deep_x]] 
            
            # Фильтрация списка #Начало [deep] или Kончало трека [deep+1]
            filtered_x0 = [num for num in level_points if (num == i_track[deep_x]) or (num == i_track[deep_x+1])]
            filtered_x1 = [num for num in level_points if (num == i_track[deep_x]) or (num == i_track[deep_x+1])]

            # Преобразуем списки в множества
            track_x = set(i_track)
            fined_x0 = set(filtered_x0)
            fined_x1 = set(filtered_x1)
            
            # Находим пересечение множеств & cчитаем количество совпадающих чисел

            counter_x0 += len(fined_x0.intersection(track_x))
            counter_x1 += len(fined_x1.intersection(track_x))
            
            # selected for consol test
            #if v == 0 or v == 4:
                #print('i_track', i_track)
                #print(f"Track{v+1}:{deep_x}-----:{counter_x0+counter_x1}")
                #print('level_points', level_points)
                #print('start', filtered_x0)
                #print('end', filtered_x1)
        
        #print("counter", counter)
        percent = (counter_x0 * 2 + counter_x1) / len(i_track) * track_max_deep * 100
        match_percents.append(percent)
    # результаты
    #for v, percent in enumerate(match_percents):
        #print(f"Трек {v+1}: вероятность {percent:.2f}%")
    return match_percents
#-------------------------------------------------------------------------------