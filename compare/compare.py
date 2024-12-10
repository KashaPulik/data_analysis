import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из первого файла
data1 = pd.read_csv('data.csv')

# Фильтрация данных для анализа температуры из первого файла
temperature_data1 = data1[['Year', 'Month', 'Temperature (°C)']]
temperature_data1['Date'] = pd.to_datetime(temperature_data1[['Year', 'Month']].assign(DAY=1))
temperature_data1.set_index('Date', inplace=True)
temperature_data1 = temperature_data1['Temperature (°C)'].resample('M').mean()

# Загрузка данных из второго файла
data2 = pd.read_csv('data1.csv')

# Фильтрация данных для анализа температуры из второго файла
temperature_data2 = data2[['Year', 'Month', 'Temperature (°C)']]
temperature_data2['Date'] = pd.to_datetime(temperature_data2[['Year', 'Month']].assign(DAY=1))
temperature_data2.set_index('Date', inplace=True)
temperature_data2 = temperature_data2['Temperature (°C)'].resample('M').mean()

# Построение графиков
plt.figure(figsize=(10, 6))

# График температуры из первого файла (синий)
plt.plot(temperature_data1.index, temperature_data1.values, label='Настоящие данные', color='blue')

# График температуры из второго файла (пунктирный)
plt.plot(temperature_data2.index, temperature_data2.values, label='Прогноз', linestyle='--', color='orange')

plt.title('Сравнение прогноза и настоящих данных')
# plt.xlabel('Дата')
# plt.ylabel('Температура, °C')
plt.legend()
plt.tight_layout()
plt.savefig('comparison.png')
