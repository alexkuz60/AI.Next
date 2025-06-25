import numpy as np

def calculate_match_percent(reference_table, data):
    V = len(data)
    rows, cols = reference_table.shape
    match_percents = []

    for v in range(V):
        table = data[v]
        matches = (table == reference_table)
        percent = np.sum(matches) / (rows * cols) * 100
        match_percents.append(percent)

    return match_percents

# Параметры
V = 50
rows = 200
cols = 10

# Набор таблиц по одному на вариант
data = [np.random.randint(0, V, size=(rows, cols)) for _ in range(V)]
# Эталонная таблица
#reference_table = np.random.randint(0, V, size=(rows, cols))
reference_table = data[3]

# Вычисляем проценты совпадения
match_percents = calculate_match_percent(reference_table, data)

# Выводим результаты
for v, percent in enumerate(match_percents):
    print(f"Вариант {v}: совпадение {percent:.2f}%")