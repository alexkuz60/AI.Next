import numpy as np
import pandas as pd

def compute_feature_matrix_numeric(V1_col, V2_col, x, X_max):
    """
    Вычисляет матрицу характеристик 3x6 для пары волн в точке x.
    Вместо бинарных признаков записывает числовые значения:
    - Для равенства (-1,0,1) записывает значение.
    - Для >1 или <-1 записывает на сколько больше или меньше.
    - Иначе np.nan.
    """
    idxs = [max(x-1,0), x, min(x+1, X_max-1)]
    vals_V1 = V1_col[idxs]
    vals_V2 = V2_col[idxs]
    
    feat_mat = np.full((3,6), np.nan)
    
    def classify_val_num(v):
        if v > 1:
            return v - 1  # насколько больше 1
        elif v == 1 or v == 0 or v == -1:
            return v     # равенство
        elif v < -1:
            return v + 1  # насколько меньше -1 (отрицательное число)
        else:
            return np.nan
    
    for i, (v1, v2) in enumerate(zip(vals_V1, vals_V2)):
        c1 = classify_val_num(v1)
        c2 = classify_val_num(v2)
        
        # Записываем значения для В1
        feat_mat[0, i] = c1 if c1 > 1 else np.nan  # строка 0: >1 (теперь числовое значение >0)
        feat_mat[1, i] = c1 if c1 in [-1,0,1] else np.nan  # строка 1: равенство
        feat_mat[2, i] = c1 if c1 < -1 else np.nan  # строка 2: < -1
        
        # Записываем значения для В2
        feat_mat[0, i+3] = c2 if c2 > 1 else np.nan
        feat_mat[1, i+3] = c2 if c2 in [-1,0,1] else np.nan
        feat_mat[2, i+3] = c2 if c2 < -1 else np.nan
    
    return feat_mat

# Остальной код без изменений, только заменяем вызов compute_feature_matrix на compute_feature_matrix_numeric

class FeatureMatrixDictionary:
    def __init__(self):
        self.matrices = []
        self.names = []
    
    def get_or_add(self, matrix):
        for i, m in enumerate(self.matrices):
            if np.array_equal(m, matrix, equal_nan=True):
                return self.names[i]
        name = f'M_{len(self.matrices)+1}'
        self.matrices.append(matrix)
        self.names.append(name)
        return name

def analyze_wave_pairs_numeric(T1, T2, markers):
    X_max, N = T1.shape
    num_markers = len(markers)
    
    dict_matrices = FeatureMatrixDictionary()
    map_table = pd.DataFrame(index=range(N), columns=range(num_markers), dtype=object)
    
    for col in range(N):
        V1_col = T1[:, col]
        V2_col = T2[:, col]
        for m_idx, (x_coord, y1_val, y2_val) in enumerate(markers):
            feat_mat = compute_feature_matrix_numeric(V1_col, V2_col, x_coord, X_max)
            name = dict_matrices.get_or_add(feat_mat)
            map_table.iat[col, m_idx] = name
    
    stats = pd.DataFrame(index=map_table.index)
    unique_names = dict_matrices.names
    for name in unique_names:
        stats[name] = (map_table == name).sum(axis=1)
    
    return map_table, dict_matrices, stats

# Пример использования:
if __name__ == "__main__":
    X_max = 30
    N = 10
    np.random.seed(0)
    T1 = np.random.randint(-3,4,size=(X_max,N))
    T2 = np.random.randint(-3,4,size=(X_max,N))
    markers = [(1,0,0), (5,0,0), (7,0,0), (12,0,0), (19,0,0), (22,0,0)]
    
    map_table, dict_matrices, stats = analyze_wave_pairs_numeric(T1, T2, markers)
    
    print("Карта-таблица матриц характеристик:")
    print(map_table)
    for i in range(len(dict_matrices.matrices)):
        print(f'-----(M{i})-----')
        print(dict_matrices.matrices[i])
    print("\nУникальные матрицы характеристик (количество):", len(dict_matrices.matrices))
    print("\nСтатистика вхождений матриц по строкам:")
    print(stats)