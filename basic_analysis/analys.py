import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

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
axes[0].plot(temperature_data, label='Временной ряд')
axes[0].set_title('Временной ряд')
axes[0].legend()

# График тренда
axes[1].plot(trend, label='Тренд', color='orange')
axes[1].set_title('Тренд')
axes[1].legend()

# График сезонности
axes[2].plot(seasonal, label='Сезонность', color='green')
axes[2].set_title('Сезонность')
axes[2].legend()

# График ошибки
axes[3].plot(residual, label='Ошибки', color='red')
axes[3].set_title('Ошибки')
axes[3].legend()

plt.tight_layout()
plt.savefig('temperature_analysis.png')
