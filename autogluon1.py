import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

data_csv = pd.read_csv("Arhiv-4x20_1.csv", delimiter='\t', encoding='utf-8')
data_csv.columns = data_csv.columns.str.strip()
data_csv['timestamp'] = pd.to_datetime(data_csv['timestamp'] + ' ' + data_csv['time'])
data_csv = data_csv.drop(columns=['time'])
data_csv['item_id'] = 1  # если один ряд

data = TimeSeriesDataFrame.from_data_frame(data_csv)

# Приводим к регулярной частоте (например, часовой)
data = data.convert_frequency('1H')

prediction_length = 8
train_data, test_data = data.train_test_split(prediction_length)

predictor = TimeSeriesPredictor(target="ord0", prediction_length=prediction_length, freq='1H').fit(train_data)

y_pred = predictor.predict(test_data)
y_pred.head()
predictor.leaderboard(test_data)

