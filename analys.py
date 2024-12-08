import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
import numpy as np

# Загрузка данных
data = pd.read_csv('data.csv')

# Фильтрация данных для анализа температуры
temperature_data = data[['Year', 'Month', 'Temperature (°C)']]
temperature_data['Date'] = pd.to_datetime(temperature_data[['Year', 'Month']].assign(DAY=1))
temperature_data.set_index('Date', inplace=True)
temperature_data = temperature_data['Temperature (°C)'].resample('M').mean()

# Разложение временного ряда
decomposition = seasonal_decompose(temperature_data, model='additive')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Построение графиков
fig, axes = plt.subplots(4, 1, figsize=(10, 15))

# График температуры
axes[0].plot(temperature_data, label='Temperature')
axes[0].set_title('Temperature')
axes[0].legend()

# График тренда
axes[1].plot(trend, label='Trend', color='orange')
axes[1].set_title('Trend')
axes[1].legend()

# График сезонности
axes[2].plot(seasonal, label='Seasonality', color='green')
axes[2].set_title('Seasonality')
axes[2].legend()

# График ошибки
axes[3].plot(residual, label='Residual', color='red')
axes[3].set_title('Residual')
axes[3].legend()

plt.tight_layout()
plt.savefig('temperature_analysis.png')

# Прогнозирование с использованием рекурсивного метода
# Подготовка данных для прогнозирования
X = np.arange(len(temperature_data)).reshape(-1, 1)
y = temperature_data.values

# Обучение модели
model = LinearRegression()
model.fit(X, y)

# Прогнозирование
future_dates = np.arange(len(temperature_data), len(temperature_data) + 12).reshape(-1, 1)
future_predictions = model.predict(future_dates)

# Построение графика прогноза
plt.figure(figsize=(10, 6))
plt.plot(temperature_data.index, temperature_data.values, label='Actual Temperature')
plt.plot(pd.date_range(start=temperature_data.index[-1], periods=12, freq='M'), future_predictions, label='Forecast', linestyle='--')
plt.title('Temperature Forecast')
plt.legend()
plt.savefig('temperature_forecast.png')
