import numpy as np
import locale
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import norm
import math

locale.setlocale(locale.LC_ALL, '')

# исходная выборка
data = [
    3.13470, 4.33256, 6.43879, 4.30368, 4.35623, 3.15246, 5.38967, 4.46868, 6.81088, 1.70559,
    2.73110, 3.37967, 4.87196, 6.80096, 3.42337, 2.38444, 6.05913, 2.45543, 2.03301, 6.57000,
    7.01645, 3.71879, 5.15612, 2.77753, 6.85072, 4.88117, 4.23474, 3.56825, 6.04606, 2.53114,
    6.93189, 5.99802, 2.78643, 3.76537, 3.48054, 5.85774, 3.21087, 4.42832, 6.51297, 3.67303,
    4.31164, 2.40515, 5.61961, 2.51263, 2.28149, 3.35978, 1.76760, 5.18527, 6.61917, 7.28909,
    6.45204, 3.71776, 3.61304, 3.24887, 7.15189, 2.17553, 4.27060, 6.38578, 3.63163, 3.26217,
    4.68601, 6.91372, 7.45664, 6.68168, 7.41742, 6.41500, 6.75989, 3.84912, 2.67972, 7.27420,
    4.42444, 3.88123, 7.45334, 1.57180, 4.38654, 5.68719, 6.19577, 3.60986, 1.60988, 6.76844,
    7.26583, 2.50760, 4.08116, 5.50057, 3.57510, 6.12859, 1.59814, 1.94384, 5.26200, 5.85418,
    2.00632, 2.01505, 4.97421, 6.07731, 1.60681, 7.33026, 5.36505, 2.19817, 1.55160, 2.54484,
    7.26704, 7.39588, 4.32885, 4.30037, 6.83677, 3.93624, 6.10259, 4.31404, 6.89171, 1.97905,
    2.54358, 7.29140, 2.35808, 5.78653, 3.84493, 2.52175, 2.24139, 5.00185, 3.72347, 1.67495,
    2.05329, 4.30598, 2.43185, 4.10002, 1.92517, 3.66897, 7.26366, 2.76846, 2.30993, 4.05263,
    3.68418, 6.15227, 4.97761, 5.99939, 4.56250, 1.83652, 2.98139, 6.62127, 6.22566, 6.02820,
    2.04173, 3.29852, 3.73881, 1.63332, 5.52920, 3.68902, 6.88417, 5.34948, 6.73851, 2.00472,
    2.20107, 3.63252, 2.43283, 2.01892, 2.48801, 4.89829, 1.72164, 1.91646, 3.15983, 7.42877,
    2.82120, 3.46002, 4.20049, 2.51770, 3.90014, 5.85073, 3.76918, 2.45202, 2.04034, 4.26143,
    4.46819, 2.12165, 3.15582, 6.83603, 7.31410, 7.21383, 3.01638, 7.30698, 7.53242, 5.09062,
    5.56761, 4.71154, 2.53161, 7.03016, 2.64399, 3.94881, 2.99900, 5.07788, 1.69509, 2.88162,
    6.74531, 1.80325, 2.99462, 5.88926, 5.93526, 6.33580, 7.32092, 7.37830, 7.36686, 6.89875
]

print("\ndata \n", data)

# отсортированная выборка
data_sorted = sorted(data)
print("\ndata sorted \n", data_sorted)

# размер выборки
N = len(data) #200
print(N)

# исходные значения
a = 1.54
b = 7.54
alf = 0.05
kalf = 1.358099

# число интервалов по формуле Стерджеса m
m = 1 + int(np.log2(N))

# a0 = min(data)
a_0 = min(data_sorted)

# am = max(data)
a_m = max(data_sorted)

# h = (am - a0) / m
h = (a_m - a_0) / m

# FN(x)    FN(x)=FN(yj) j =1..(m-1)
FNy = np.zeros(N+1)
for j in range(1, N+1):
    FNy[j] = j / N

def FNX(x):
    for j in range(1, N+1):
        if (data_sorted[j-1] <= x < data_sorted[j]):
            FNx = FNy[j]
            break
        elif (x < data_sorted[j-1]) and (j == 1):
            FNx = 0
            break
        elif (x >= data_sorted[N-1]):
            FNx = 1
            break
    return FNx
    
#FN0 = (j - 1) / N
FNx0 = 0
def FNX0(x):
    global FNx0
    for j in range(0, N):
        if data_sorted[j] == x:
            FNx0 = (j+1 - 1) / N
            #print(j+1)
    return FNx0

L = [FNX0(x) for x in data_sorted]
# эмпирическая функция Fx
def uniform_cdf(x):
    return (x - a) / (b - a) if a <= x <= b else (0 if x < a else 1)


# DN
J = 0
MAX1 = np.zeros(N+1)
for j in range(1, N+1):
    MAX1[j]= max(np.abs(FNy[j] - uniform_cdf(data_sorted[j-1])), np.abs(FNX0(data_sorted[j-1]) - uniform_cdf(data_sorted[j-1])))

DN = max(MAX1)
J = next(i for i, value in enumerate(MAX1) if value == DN) 

# DNsqrt(N)
DNN = round(DN * np.sqrt(N),5)

# y* = yy
MAX1y = 0
for j in range(1, N+1):
    if DN == MAX1[j]:
        yy = data_sorted[j-1]

# F(y*)
Fyy = uniform_cdf(yy)

# FN(y*)
FNyy = FNX(yy)

# FN(y*-0)
FNyy0 = FNX0(yy)

data_filtered = [a] + [x for x in data_sorted if a < x < b] + [b]

N_filtered = len(data_filtered)
intervalss = data_filtered[0] + np.arange(m) * h  
intervalss = np.append(intervalss, b)
FNx0_x = data_filtered
FNx0_y = [FNX0(x) for x in FNx0_x]

print("\nТаблица 4.3:\n")
print("|  a   |  b   |  N  |    DN   | DNsqrt(N) |    y*   |  F(y*)  |  FN(y*)  |  FN(y*-0)  |")
print("|------|------|-----|---------|-----------|---------|---------|----------|------------|")

# Заполнение таблицы
print(f"| {a:<3.2f} | {b:<3.2f} | {N:<3} | {DN:<7.5f} |  {DNN:<9.5f}| {yy:<6.5f} | {Fyy:<7.5f} | {FNyy:<8.5f} |  {FNyy0:<9.5f} |")
print("|------|------|-----|---------|-----------|---------|---------|----------|------------|")


print("\nПроверка гипотезы с помощью критерия Колмогорова")
if DNN <= kalf:
    print(f"\n{DNN:.5f} <= {kalf} Гипотеза о соответствии выборки заданному распределению не противоречит экспериментальным данным при уровне значимости a = 0.05")
elif DNN > kalf:
    print(f"\n{DNN:.5f} > {kalf} Гипотеза о соответствии выборки заданному распределению противоречит экспериментальным данным при уровне значимости a = 0.05")

ecdf_x = data_filtered
ecdf_y = [uniform_cdf(x) for x in ecdf_x]

# Построение графика
plt.figure(figsize=(10, 6))

for i in range(len(FNx0_x)):
    if i == 0:
        plt.hlines(FNx0_y[i], xmin=FNx0_x[i], xmax=FNx0_x[i], linewidth=2, color='blue', label='Эмпирическая функция FN(x)' )  
    else:
        plt.hlines(FNx0_y[i], xmin=FNx0_x[i-1], xmax=FNx0_x[i], linewidth=2)

plt.plot(ecdf_x, ecdf_y, color='red', label=' Функция распределения F(x)', linewidth=2)

plt.xticks(intervalss)
plt.xlim(a- 0.2, b+0.2)  
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Эмпирическая и функция распределения')
plt.grid(True)
plt.legend()
plt.show()


