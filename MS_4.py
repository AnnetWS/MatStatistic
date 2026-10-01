import numpy as np
import locale
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
import math

locale.setlocale(locale.LC_ALL, '')
# исходная выборка
data = [
    2.18590, -1.40348, 3.62469, 2.07147, 0.48154, 4.06655, 0.54151, 3.91663, -0.61092, 3.60442,
    3.04385, 1.95526, 0.64328, 1.94153, 2.95652, 2.21761, 1.84440, 0.01477, 2.21962, 2.64431,
    2.05754, 4.33728, 3.24433, 2.42392, 2.22455, -0.18314, 1.81059, 2.88235, 3.54554, 1.82262,
    -1.77511, 3.07028, 0.74972, 1.17200, 3.33255, 4.00102, 1.91952, 1.84231, 1.37931, 3.68091,
    0.16472, 3.17544, 3.53045, 2.44281, 1.66414, 2.85028, 2.11242, 2.46313, 1.25874, 1.95125,
    1.11022, 5.12107, 3.76079, 1.73729, 2.72943, 2.95357, 5.04695, 4.44961, 1.93987, 1.89933,
    3.58406, 1.28563, 1.47032, 0.63086, 2.16834, 3.87880, 2.74821, 2.54465, 4.25881, 3.60249,
    3.44667, 0.45408, 1.79291, 3.86963, 1.22572, 1.61318, 1.48649, 2.34531, 1.55585, 3.81192,
    2.21690, 0.64354, 0.74295, 3.00913, 0.17598, 2.30379, 2.96692, 1.49066, 0.91152, 3.68608,
    2.47319, 1.13075, 1.91766, 2.05448, 3.35754, 4.60914, 1.27476, 0.13144, 2.51414, 2.99780,
    3.13718, 1.93710, 2.14912, 1.47573, 2.31967, 1.11792, 0.89324, 3.14983, 3.07592, 0.33140,
    -0.82444, 0.36346, 3.26917, 2.93191, 3.11892, 2.45080, 1.41725, 1.93532, 3.94556, 2.16888,
    2.71829, 0.40253, 1.01129, 0.52005, 2.96763, 0.09103, 1.13175, 3.49969, 3.37718, 0.95550,
    2.21166, 1.02302, 0.74734, 1.10303, 3.19147, 3.80812, 2.53740, 3.57293, 1.77523, 1.37433,
    2.30583, 1.60535, 1.21696, 2.75022, 0.34824, 4.13993, 2.07676, 2.55194, 2.54196, 1.79090,
    3.13251, 3.47970, 3.00590, -0.49827, -1.03014, 1.95355, 3.22079, 3.11076, 2.44583, 4.01999,
    -0.07119, 3.11588, 3.35392, 3.42735, 2.05323, 1.29702, 0.46180, -1.32829, 1.53378, 2.23963,
    -0.65244, 2.98434, 2.96742, 0.50136, 2.85549, 1.78673, 4.13147, 2.41131, 5.15599, 1.76785,
    3.57392, 4.11506, 4.02831, 0.26683, 2.53379, 2.44978, 1.72128, 0.02585, 0.77420, 2.47149,
    3.70511, 2.97380, 1.41070, 3.02451, 1.33764, -0.36536, 0.86172, 0.16376, 1.79294, 2.85804
]
print("\ndata \n", data)

# отсортированная выборка
data_sorted = sorted(data)
print("\ndata sorted \n", data_sorted)

# размер выборки
N = len(data) #200
print(N)

# число интервалов по формуле Стерджеса m
m = 1 + int(np.log2(N))
print("\nЧисло интервалов m = ", m)

# a0 = min(data)
a_0 = min(data_sorted)
print("\na_0 = ", a_0)

# am = max(data)
a_m = max(data_sorted)
print("\na_m = ", a_m)

# h = (am - a0) / m
h = (a_m - a_0) / m
print("\nh = ", h)

# вывод интервалов (ak-1, ak]
intervals = np.linspace(a_0, a_m, m + 1)
intervals_list = np.array([(intervals[i], intervals[i + 1]) for i in range(m)])

# Подсчет количества значений в каждом интервале и относительных частот n_i, w_i
n_i = [int(np.sum((data_sorted >= intervals[i]) & (data_sorted <= intervals[i+1]))) for i in range(m)]
w_i = np.array(n_i) / N 

sum_w_i = np.sum(w_i)
sum_n_i = np.sum(n_i)

# интервальный ряд таблица
max_interval_length = max(len(f"({intervals[i]:.5f}, {intervals[i + 1]:.5f}]") for i in range(m))
max_n_i_length = max(len(str(n)) for n in n_i)
max_w_i_length = max(len(f"{w:.2f}") for w in w_i)

# Вывод таблицы
print("\nТаблица 4.1:")
print("\n| Интервалы              | n_i  |  w_i      |")
print("|------------------------|------|-----------|")

# Заполнение таблицы
for k in range(1,m + 1):
     start, end = intervals_list[k - 1]
     print(f"| ({start:<8.5f}, {end:<8.5f})   |  {n_i[k-1]: <3} | {w_i[k-1]:<9.5f} |")

print("|------------------------|------|-----------|")
print(f"|------------------------| {sum_n_i:<3}  |    {sum_w_i:<3}    |")

# мат ожидание
total_sum = sum(data)
MO = (1/N) * total_sum
print(f"\nОценка Мат ожидание = {MO:.5f}")

# дисперсия
total_sum_d = sum(x**2 for x in data)
disp = (1/N) * total_sum_d - (MO**2) - ((h**2)/12)
print(f"\nОценка Дисперсия = {disp:.5f}")

# среднее квадратическое отклонение
std_deviation = disp**0.5
print(f"\nОценка Среднее квадратическое отклонение = {std_deviation:.5f}")

# k = 0..m
K = np.zeros(m + 1)
for k in range(0, m+1):
    K[k] = k

# ak - a^ / q
s = np.zeros(m + 1)
for k in range(0, m+1):
    s[k] = (intervals[k] - MO) / std_deviation

# функция плотности стандартного нормального распределения f0
def standard_normal_pdf(t):
    return (1 / np.sqrt(2 * np.pi)) * np.exp((-(t**2)) / 2)

# Ф(x) функция распределения стандартного нормального закона
def standard_normal_cdf(x):
    # Выполнение интегрирования от -бесконечности до x
    #integral, _ = quad(standard_normal_pdf, -np.inf, x)
    integral = norm.cdf(x)
    return integral

# 1/f0(pdf) * pdf
D = np.zeros(m + 1)
for k in range(0, m+1):
    count = 1/std_deviation
    D[k] = count * standard_normal_pdf(s[k])
    print("D = ", D)


# pk*
def pk(k, p): # k = 2 .. (m-1)
    if k == 1:
        p[k] = standard_normal_cdf(s[1])
    elif k == m:
        p[k] = 1 - standard_normal_cdf(s[m-1])
    elif 2 <= k <= (m - 1):
        p[k] = standard_normal_cdf(s[k]) - standard_normal_cdf(s[k-1])
    return p

p = np.zeros(m + 1)
for k in range(1, m+1):
    p = pk(k, p)

sum_pk = np.sum(p)

# Создаем таблицу
print("\nТаблица 4.2:\n")
print("| K |      a_k       |  a_k-a^/d^ | 1/d^*f0(a_k-a^/d^) |    Ф(a_k-a^/d^)    |   p_k*   |")
print("|---|----------------|------------|--------------------|--------------------|----------|")

# Заполнение таблицы
for k in range(0,m + 1):
     print(f"| {k: <1} | {intervals[k]: <14.5f} | {s[k]: <10.5f} | {D[k]: <18.5f} | {standard_normal_cdf(s[k]): <18.5f} | {p[k]: <8.5f} |")

print("|---|----------------|------------|--------------------|--------------------|----------|")
print(f"|---|----------------|------------|--------------------|--------------------|   {sum_pk:<3}    |")

# |wk - pk|
WPk = np.zeros(m + 1)
for i in range(1, m + 1):
    WPk[i] = round(np.abs(w_i[i-1] - p[i]),5)

max_WPk = np.max(WPk[1:]) 
index_max_WPk = np.argmax(WPk[1:]) + 1  


# N(WPk)**2/pk
XV = np.zeros(m + 1)
for i in range(1, m + 1):
    F = w_i[i-1] - p[i]
    XV[i] = (N * (F**2))/ p[i]

sum_XV = np.sum(XV)

print("\nТаблица 4.3:\n")
print("| K | Интервалы              | w_i        | p_k        |   |wk - pk|   | N(wk - pk)/pk|")
print("|---|------------------------|------------|------------|---------------|--------------|")

# Заполнение таблицы
for k in range(1,m + 1):
     start, end = intervals_list[k - 1]
     print(f"| {k: <1} | ({start:<8.5f}, {end:<8.5f})   | {w_i[k-1]: <10.5f} | {p[k]: <10.5f} | {WPk[k]: <13.5f} | {XV[k]: <13.5f}|")

print("|---|------------------------|------------|------------|---------------|--------------|")
print(f"|---|------------------------|    {sum_w_i:<3}     |  {sum_pk:<9} | max = {max_WPk:<3.5f} | {sum_XV:<13.5f}|")


# Проверить с помощью критерия гипотезу о соответствии выборки нормальному распределению
l = m - 3

Xkp = 10.191028
result = D[4]/h
print("res ", result)

print("\nПроверка гипотезы с помощью критерия")
if sum_XV <= Xkp:
    print(f"\n{sum_XV:.5f} <= {Xkp} Гипотеза о соответствии выборки заданному распределению не противоречит экспериментальным данным при уровне значимости a = 0.07")
elif sum_XV > Xkp:
    print(f"\n{sum_XV:.5f} > {Xkp} Гипотеза о соответствии выборки заданному распределению противоречит экспериментальным данным при уровне значимости a = 0.07")

plt.figure(figsize=(9, 5))
plt.bar(intervals[:-1], height= (w_i/h), width=h, align='edge', alpha=0.6, label='Гистограмма относительных частот')
x = np.linspace(a_0, a_m, 1000)
pdf = norm.pdf(x, MO, std_deviation)
plt.plot(x, pdf, 'r-', linewidth=2, label='Плотность нормального распределения')

plt.xlabel('x')
plt.ylabel('Относительные частоты')
plt.title('Гистограмма относительных частот')
plt.ylim(0.0, 0.4) 
plt.xticks(intervals)
plt.grid(True)
plt.legend()
plt.show()