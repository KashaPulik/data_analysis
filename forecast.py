import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Загрузка данных
data = pd.read_csv('data.csv')

# Фильтрация данных для анализа температуры
temperature_data = data[['Year', 'Month', 'Temperature (°C)']]
temperature_data['Date'] = pd.to_datetime(temperature_data[['Year', 'Month']].assign(DAY=1))
temperature_data.set_index('Date', inplace=True)
temperature_data = temperature_data['Temperature (°C)'].resample('M').mean()

# Обучение модели SARIMA
model = SARIMAX(temperature_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Прогнозирование
forecast = results.get_forecast(steps=12)
forecast_ci = forecast.conf_int()

# Построение графика прогноза
plt.figure(figsize=(10, 6))
plt.plot(temperature_data.index, temperature_data.values, label='Actual Temperature')
plt.plot(forecast.predicted_mean.index, forecast.predicted_mean, label='Forecast', linestyle='--')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='k', alpha=.2)
plt.title('Temperature Forecast using SARIMA')
plt.legend()
plt.savefig('temperature_forecast_sarima.png')
