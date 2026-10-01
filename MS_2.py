import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom
import math
import pandas as pd
import locale

locale.setlocale(locale.LC_ALL, '')
# Параметры
V = 134
N = 200
#n = 5 + V % 20
p = 0.2 + 0.003 * V
q = 1 - p
print("\np = ",f"{p:.5f}")
print("\nq = ",f"{q:.5f}")
# Генерация выборки +++
sample = geom.rvs(p, size=N)

# Неупорядоченная выборка +++
print("Неупорядоченная выборка:")
print(sample)

# Упорядоченная выборка +++
sorted_sample = np.sort(sample)
print("\nУпорядоченная выборка:")
print(sorted_sample)

x_sorted = np.sort(np.unique(sample))
n = len(sample)
w_values = np.bincount(sample)[x_sorted] / n

# 1. Статистический ряд +++
unique, counts = np.unique(sample, return_counts=True)
frequencies = dict(zip(unique, counts))
relative_frequencies = {k: v / N for k, v in frequencies.items()}
cumulative_frequencies = {k: sum(v for key, v in relative_frequencies.items() if key <=k) for k in sorted(relative_frequencies.keys())}


statistical_series = []
for x in sorted(frequencies.keys()):
    statistical_series.append([x, frequencies[x], relative_frequencies[x], cumulative_frequencies[x]])

print("Статистический ряд:")
print("{:<5} {:<5} {:<10} {:<10}".format("x_i", "n_i", "w_i", "s_i"))
for row in statistical_series:
    print("{:<5} {:<5} {:<10.5f} {:<10.5f}".format(row[0], row[1], row[2], row[3]))
print("-" * 30)
print("{:<5} {:<5} {:<10.5f} {:<10}".format("", sum(row[1] for row in statistical_series), sum(row[2] for row in statistical_series), ""))

n_i_array = np.array([row[1] for row in statistical_series])

# 2. Полигон относительных частот и теоретических вероятностей +++
x_values = np.array(list(frequencies.keys()))
w_values = np.array(list(relative_frequencies.values()))
theoretical_probabilities = geom.pmf(x_values, p)

# сортировка данных для правильного построения полигона
sorted_indices = np.argsort(x_values)
x_values_sorted = x_values[sorted_indices]
w_values_sorted = w_values[sorted_indices]
theoretical_probabilities_sorted = theoretical_probabilities[sorted_indices]

plt.figure(figsize=(10, 6))
plt.plot(x_values_sorted, w_values_sorted, marker='o', linestyle='-', label='Относительные частоты')
plt.plot(x_values_sorted, theoretical_probabilities_sorted, marker='x', linestyle='-', color='red', label='Теоретические вероятности')
plt.xlabel('x_i')
plt.ylabel('Частота/Вероятность')
plt.title('Полигон относительных частот и теоретических вероятностей')
plt.legend()
plt.grid(True)


# 3. Эмпирическая функция распределения +++
x_values = x_sorted  

ecdf_values = np.zeros_like(x_values, dtype=float)
cumulative_sum = 0
x_sorted_index = 0
for i, x in enumerate(x_values):
    while x_sorted_index < len(x_sorted) and x >= x_sorted[x_sorted_index]:
        cumulative_sum += w_values[x_sorted_index]
        x_sorted_index += 1
    ecdf_values[i] = cumulative_sum

diffs = np.diff(x_sorted)
offsets = np.concatenate((diffs, [1])) 

x_values_spread = x_sorted + offsets

plt.figure(figsize=(9, 5))

for i, x in enumerate(x_values_spread): 
    plt.hlines(ecdf_values[i], x_sorted[i], x, linewidth=2) 

plt.xticks(x_sorted)
plt.xlabel('x')
plt.ylabel('F_N(x)')
plt.title('Эмпирическая функция распределения')
plt.ylim(-0.1,1.1)
plt.grid(True)

# Найти:

# 1.Выборочное среднее +++
x_values = np.array(list(frequencies.keys()))
w_values = np.array(list(relative_frequencies.values()))

sample_mean = 0
for i in range(len(x_values)):
    sample_mean += (float(x_values[i]) * w_values[i])
 
print("\nВыборочное среднее:", f"{sample_mean:.5f}") 

# 2.Выборочная дисперсия +++
variance = 0.0
for i in range(len(x_values)):
    variance += ((x_values[i] - sample_mean)**2) * w_values[i]
print("\nВыборочная дисперсия:", f"{variance:.5f}") 

# Выборочный центральный момент к-ого порядка +++
mu1 = 0.0
mu2 = 0.0
mu3 = 0.0
mu4 = 0.0
for i in range(len(x_values)):
    mu1 += ((x_values[i] - sample_mean)**1) * w_values[i]
    mu2 += ((x_values[i] - sample_mean)**2) * w_values[i]
    mu3 += ((x_values[i] - sample_mean)**3) * w_values[i]
    mu4 += ((x_values[i] - sample_mean)**4) * w_values[i]
print("\nВыборочный центральный момент mu1:", f"{mu1:.5f}") 
print("\nВыборочный центральный момент mu2:", f"{mu2:.5f}") 
print("\nВыборочный центральный момент mu3:", f"{mu3:.5f}") 
print("\nВыборочный центральный момент mu4:", f"{mu4:.5f}") 

# 3.Выборочное среднее квадратическое отклонение +++
std_dev = 0.0
std_dev = np.sqrt(variance)
print("\nВыборочное среднее квадратическое отклонение:", f"{std_dev:.5f}") 

# 4.Выборочная мода +++
sample_mode = 0
max_freq = np.max(n_i_array)
max_indices = np.where(n_i_array == max_freq)[0]
if len(max_indices) == 1:  
    if max_indices == 0:
        sample_mode = 0
    else:
        sample_mode = x_values[max_indices[0]]
elif np.all(np.diff(max_indices) == 1): 
    sample_mode = 0.5 * (x_values[max_indices[0]] + x_values[max_indices[-1]])
else:  
    sample_mode = 0

print("\nВыборочная мода:", f"{sample_mode:.5f}")

# 5.Выборочная медиана +++
median = 0.0
n = len(x_sorted)
print("ecdf:", ecdf_values)
print("n:", n)
for i in range(n):
    if ecdf_values[i] >= 0.5:
        if ecdf_values[i] == 0.5 and i + 1 < n:
            median = (x_sorted[i] + x_sorted[i+1]) / 2
        else:
            median = x_sorted[i]
        break
print("\nВыборочная медиана:", f"{median:.5f}")

# 6.Выборочный коэффициент асимметрии +++
sample_skewness = 0.0
sample_skewness = mu3/(std_dev**3)
print("\nВыборочный коэффициент асимметрии", f"{sample_skewness:.5f}")

# 7.Выборочный коэффициент эксцесса +++
sample_kurtosis = 0.0
sample_kurtosis = mu4/(std_dev**4) - 3
print("\nВыборочный коэффициент эксцесса", f"{sample_kurtosis:.5f}")

def calculate_theoretical_characteristics(n, p):
    mean = q / p
    variance = q / p**2
    std_dev = np.sqrt(q) / p

    mode = 0

    summ = np.cumsum(theoretical_probabilities)
    median = np.argmax(summ >= 0.5) + 1  

    skew = (1 + q) / np.sqrt(q)
    kurt = 6 + (p**2/q)

    return mean, variance, std_dev, mode, median, skew, kurt

# составить таблицы:
# 1) сравнения относительных частот и теоретических вероятностей;
x_values = np.array(list(frequencies.keys()))
w_values = np.array(list(relative_frequencies.values()))
theoretical_probabilities = geom.pmf(x_values+1, p)

# Подготовка данных для таблицы
data = {
    'x_i': x_values,
    'w_i': w_values,
    'p_i*': theoretical_probabilities,
    '|w_i - p_i*|': np.abs(w_values - theoretical_probabilities)
}
df = pd.DataFrame(data)
summary_data = {
    'x_i': ['   '],
    'w_i': [df['w_i'].sum()],
    'p_i*': [df['p_i*'].sum()],
    '|w_i - p_i*|': [df['|w_i - p_i*|'].max()]
}
summary_df = pd.DataFrame(summary_data)

final_df = pd.concat([df, summary_df], ignore_index=True)
print("\nТаблица сравнения:")
print(final_df.round(5))

# 2) сравнения рассчитанных характеристик с теоретическими значениями.
theoretical_mean, theoretical_variance, theoretical_std_dev, theoretical_mode, theoretical_median, theoretical_skew, theoretical_kurt = calculate_theoretical_characteristics(n, p)
# Подготовка данных для таблицы
data = {
    'Название': ['Среднее', 'Дисперсия', 'Среднее кв откл', 'Мода', 'Медиана', 'асимметрия', 'эксцесса'],
    'Выборочное': [sample_mean, variance, std_dev, sample_mode, median, sample_skewness, sample_kurtosis],
    'Теоретич': [theoretical_mean, theoretical_variance, theoretical_std_dev, theoretical_mode, theoretical_median, theoretical_skew, theoretical_kurt]
}
df = pd.DataFrame(data)

df['Абсолют о'] = abs(df['Выборочное'] - df['Теоретич'])
df['Абсолют о'] = df['Абсолют о'].round(5)
df['Относит о'] = np.where(df['Теоретич'] != 0, (df['Абсолют о'] / abs(df['Теоретич'])).round(5), np.nan) 
df['Относит о'] = df['Относит о'].fillna('-') 

print("\nТаблица сравнения рассчитанных характеристик с теоретическими значениями:")
print(df.round(5))

plt.show()