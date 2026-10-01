import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import masked_print_option
from pandas.tseries.offsets import LastWeekOfMonth
from scipy import stats
import locale
import pandas as pd

locale.setlocale(locale.LC_ALL, '')

# Параметры
V = 134
N = 200
lambda_param = 1 + (-1)**V * 0.003 * V
print("\nlambda_param = ", f"{lambda_param:.5f}")

# Генерация экспоненциальной случайной выборки +++
sample = np.random.exponential(scale = (1/lambda_param), size=N)

np.set_printoptions(precision=5, suppress=True)
# Вывод неупорядоченной выборки
print("Неупорядоченная выборка:\n", sample)

# Вывод упорядоченной выборки
sample_sorted = np.sort(sample)
print("\nУпорядоченная выборка:\n", sample_sorted)

# Уникальные значения из выборки
x_i_values_unique = np.unique(sample)
print("\nxvaluesunique:", x_i_values_unique)

# Определение количества интервалов по формуле Стерджеса
m = 1 + int(np.log2(N))
print("\n m(количество интервалов): ", m)

# Определение границ интервалов
a_0 = 0.0
a_m = sample_sorted.max()
interval_width = (a_m - a_0) / m
intervals = a_0 + np.arange(m + 1) * interval_width
print("\n intervals:", intervals)

# Подсчет количества значений в каждом интервале и относительных частот
n_i = []
w_i = []
for i in range(m):
    count = np.sum((sample_sorted > intervals[i]) & (sample_sorted <= intervals[i+1]))
    n_i.append(count)
    w_i = np.array(n_i) / N 

# Вычисление середин интервалов
x_i_star = np.round((intervals[:-1] + intervals[1:]) / 2, 5) 

np.set_printoptions(precision=5, suppress=True)

# построить:

# 1) график эмпирической функции распределения 
sorted_sample = np.sort(sample)
ecdf_x = sorted_sample
ecdf_y = np.arange(1, N + 1) / N

plt.figure(figsize=(10, 6))

for i in range(len(ecdf_x)):
    if i == 0:
        plt.hlines(ecdf_y[i], xmin=ecdf_x[i], xmax=ecdf_x[i], linewidth=2)  
    else:
        plt.hlines(ecdf_y[i], xmin=ecdf_x[i-1], xmax=ecdf_x[i], linewidth=2) 

plt.xticks(intervals)
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Эмпирическая функция распределения')
plt.grid(True)

# 2) интервальный ряд 
interval_series = []
for i in range(m):
    interval_series.append([f"[{intervals[i]:.5f},{intervals[i+1]:.5f}]", n_i[i], w_i[i]])

print("Интервальный ряд:")
print("Интервалы       n_i      w_i")
for row in interval_series:
  print(f"{row[0]:15s} {row[1]:6d}  {row[2]:.5f}")

print(f"\nСумма n_i: {np.sum(n_i)}")
print(f"Сумма w_i: {np.sum(w_i):.5f}")

# 2) ассоциированный статистический ряд
statistical_series = np.column_stack((x_i_star, n_i, w_i))

print("\nАссоциированный статистический ряд:")
print("x_i*     n_i     w_i")
for row in statistical_series:
    print(f"{row[0]}  {row[1]: f}  {row[2]:.5f}")

print(f"\nСумма n_i: {np.sum(n_i)}")
print(f"Сумма w_i: {np.sum(w_i):.5f}")

# 3. Гистограмма относительных частот с наложением плотности
plt.figure(figsize=(9, 5))
h =  (a_m - a_0) / m
num_intervals = len(w_i)
intervalsS = np.arange(0, num_intervals * h, h)
plt.bar(intervalsS, w_i, width=h, align='edge', alpha=0.6, label='Гистограмма относительных частот')

x = np.linspace(0, np.max(sample_sorted), 1000)
pdf = stats.expon.pdf(x, scale=1/lambda_param)
pdf_n = pdf / np.max(pdf)
plt.plot(x, pdf_n, 'r-', linewidth=2, label='Плотность Пуассона')

plt.xlabel('x')
plt.ylabel('Относительная частота / Плотность')
plt.title('Гистограмма относительных частот с кривой плотности')
plt.ylim(0.0, 1.1)
plt.xticks(intervalsS)
plt.grid(True) 

# Найти:
# 1) выборочное среднее
sample_mean = 0
for i in range(len(x_i_star)):
    sample_mean += (x_i_star[i] * w_i[i])
print("\nВыборочное среднее:", f"{sample_mean:.5f}")

# 2) выборочную дисперсию с поправкой Шеппарда
variance = 0.0
for i in range(len(x_i_star)):
    variance += ((x_i_star[i] - sample_mean)**2) * w_i[i]
variance -= h**2/12
print("\nВыборочная дисперсия cправкой Шеппарда:", f"{variance:.5f}") 

# 3) выборочное среднее квадратическое отклонение
std_dev = np.sqrt(variance)
print("\nВыборочное среднее квадратическое отклонение:", f"{std_dev:.5f}") 

# Выборочный центральный момент к-ого порядка +++
mu1 = 0.0
mu2 = 0.0
mu3 = 0.0
mu4 = 0.0
for i in range(len(x_i_star)):
    mu1 += ((x_i_star[i] - sample_mean)**1) * w_i[i]
    mu2 += ((x_i_star[i] - sample_mean)**2) * w_i[i]
    mu3 += ((x_i_star[i] - sample_mean)**3) * w_i[i]
    mu4 += ((x_i_star[i] - sample_mean)**4) * w_i[i]
print("\nВыборочный центральный момент mu1:", f"{mu1:.5f}") 
print("\nВыборочный центральный момент mu2:", f"{mu2:.5f}") 
print("\nВыборочный центральный момент mu3:", f"{mu3:.5f}") 
print("\nВыборочный центральный момент mu4:", f"{mu4:.5f}") 

# 4) выборочную моду
max_freq = np.max(w_i)
modal_indices = np.where(w_i == max_freq)[0]
k = modal_indices[0]
if len(modal_indices) == 0:
    mode = 0
elif len(modal_indices) == 1:  
    if k == 0:
        mode = 0
    elif k > 0:
        l = len(modal_indices)
        a_k = intervals[k]
        a_k_plus_1 = intervals[k + 1] if k + 1 < len(intervals) else intervals[k] + interval_width
        w_k = w_i[k - 1] if k > 0 else 0
        w_k_plus_1 = w_i[k]
        w_k_plus_2 = w_i[modal_indices[-1] + 1] if modal_indices[-1] + 1 < len(w_i) else 0
        numerator = w_k_plus_1 - w_k
        denominator = 2 * w_k_plus_1 - w_k - w_k_plus_2
        if denominator == 0:
            mode = None
        else:
            mode = a_k + h * (numerator / denominator)
elif len(modal_indices) > 1: 
    if np.all(np.diff(modal_indices) == 1): 
        l = len(modal_indices)
        a_k = intervals[k]
        a_k_plus_1 = intervals[k + 1] if k + 1 < len(intervals) else intervals[k] + interval_width
        w_k = w_i[k - 1] if k > 0 else 0
        w_k_plus_1 = w_i[k]
        w_k_plus_2 = w_i[modal_indices[-1] + 1] if modal_indices[-1] + 1 < len(w_i) else 0
        numerator = w_k_plus_1 - w_k
        denominator = 2 * w_k_plus_1 - w_k - w_k_plus_2
        if denominator == 0:
            mode = None
        else:
            mode = a_k + h * l * (numerator / denominator)
    else:
        mode = None  

print("\nВыборочная мода:", mode)

# 5) выборочную медиану
median = 0
cumulative_sum = np.cumsum(w_i)

for k in range(len(cumulative_sum)):
    if cumulative_sum[k] >= 0.5:
        break

if k == 0:
    median = 0
elif k == 1:
    median = 0
elif k > 1:
    a_k_minus = intervals[k-1]
    a_k = intervals[k]
    w_k = w_i[k - 1] 
    numerator = 0.5 - cumulative_sum[k - 1]
    median = a_k_minus + (numerator / w_k) * interval_width
print("\nВыборочная медиана:", f"{median:.5f}")

# 6) выборочный коэффициент асимметрии;
sample_skewness = mu3/(std_dev**3)
print("\nВыборочный коэффициент асимметрии:", f"{sample_skewness:.5f}")

# 7) выборочный коэффициент эксцесса;
sample_kurtosis = mu4/(std_dev**4) - 3
print("\nВыборочный коэффициент эксцесса:", f"{sample_kurtosis:.5f}")


# составить таблицы:

# 1) сравнения относительных частот и теоретических вероятностей попадания в интервалы;

theoretical_probabilities = np.array([np.exp(-lambda_param * intervals[i]) - np.exp(-lambda_param * intervals[i+1]) for i in range(m)])
intervals_list = np.array((intervals[i].round(5), intervals[i+1].round(5)) for i in range(m))
data = {
    'Interval': intervals_list,
    'wi': w_i,
    'pi': theoretical_probabilities,
    '|wi - pi|': np.abs(w_i - theoretical_probabilities)
}
df = pd.DataFrame(data)

summary_data = {
    'Interval': ['   '],
    'wi': [df['wi'].sum()],
    'pi': [df['pi'].sum()],
    '|wi - pi|': [df['|wi - pi|'].max()]
}
summary_df = pd.DataFrame(summary_data)

final_df = pd.concat([df, summary_df], ignore_index=True)
print(f"\nТаблица сравнения относительных частот и теоретических вероятностей попадания в интервалы:\n")
print(final_df.round(5))


def calculate_theoretical_characteristics(lambda_param):
    mean = 1 / lambda_param
    variance = 1 / lambda_param**2
    std_dev = 1 / lambda_param

    mode = 0

    median = (1 / lambda_param) * np.log(2)

    skew = 2
    kurt = 6 

    return mean, variance, std_dev, mode, median, skew, kurt

# 2) сравнения рассчитанных характеристик с теоретическими значениями.
theoretical_mean, theoretical_variance, theoretical_std_dev, theoretical_mode, theoretical_median, theoretical_skew, theoretical_kurt = calculate_theoretical_characteristics(lambda_param)

data = {
    'Название': ['Среднее', 'Дисперсия', 'Среднее кв откл', 'Мода', 'Медиана', 'асимметрия', 'эксцесса'],
    'Выборочное': [sample_mean, variance, std_dev, mode, median, sample_skewness, sample_kurtosis],
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