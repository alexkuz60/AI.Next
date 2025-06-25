import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

file_path = 'Arhiv-4x20.csv'

col_names = ['item_id', 'date', 'time', 'series1', 'series2', 'series3', 'series4', 'series5', 'series6', 'series7', 'series8']

# Читаем CSV
df = pd.read_csv(file_path, header=None, names=col_names)

# Объединяем дату и время в datetime
df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'], format='%d.%m.%y %H:%M')

# Удаляем лишние колонки
df = df.drop(columns=['date', 'time'])

# Один временной ряд с 8 каналами
df['item_id'] = 1

# Сортируем по времени
df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

# Создаём TimeSeriesDataFrame
data = TimeSeriesDataFrame.from_data_frame(df)

# Приводим данные к регулярной частоте 1 час с заполнением пропусков NaN
data = data.convert_frequency('1h')

# Длина прогноза — 5 часов (следующие 5 точек)
prediction_length = 5

# Разбиваем на train/test: последние 5 точек — тест
train_data, test_data = data.train_test_split(prediction_length)

print(f"Train data length: {len(train_data)}")
print(f"Test data length: {len(test_data)}")
# Проверяем типы
print(train_data.dtypes)

# Получаем pandas DataFrame из TimeSeriesDataFrame
train_df = train_data.to_data_frame()

# Приводим целевые колонки к числовым
for col in ['series1', 'series2', 'series3', 'series4', 'series5', 'series6', 'series7', 'series8']:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce')

# Создаём обратно TimeSeriesDataFrame
train_data = TimeSeriesDataFrame.from_data_frame(train_df)

# Аналогично для test_data
test_df = test_data.to_data_frame()
for col in ['series1', 'series2', 'series3', 'series4', 'series5', 'series6', 'series7', 'series8']:
    test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
test_data = TimeSeriesDataFrame.from_data_frame(test_df)
print('test data:', test_data)

# Создаём и обучаем предиктор с пресетом best_quality
predictor = TimeSeriesPredictor(
    target=['series1', 'series2', 'series3', 'series4', 'series5', 'series6', 'series7', 'series8'],
    prediction_length=prediction_length,
    freq='1h'
).fit(train_data, presets='best_quality')

# Прогнозируем на тестовом наборе
predictions = predictor.predict(test_data)

# Выводим прогнозы
print("Прогнозы:")
print(predictions.to_data_frame())

# Выводим реальные значения
print("Реальные значения:")
print(test_data.to_data_frame())

# Оцениваем качество прогноза
performance = predictor.evaluate(test_data)
print("Оценка качества по каждому ряду:")
print(performance)