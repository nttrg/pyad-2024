import numpy as np
import scipy as sc
from scipy.optimize import minimize_scalar
from scipy.stats import kurtosis, skew


def matrix_multiplication(matrix_a, matrix_b):
    """
    Задание 1. Функция для перемножения матриц с помощью списков и циклов.
    Вернуть нужно матрицу в формате списка.
    """
    if len(matrix_a[0]) != len(matrix_b):
        raise ValueError("Умножение выполнить невозможно, так как число столбцов матрицы A и число строк матрицы B не равны.")

    m = len(matrix_a)  # кол-во строк в матрице A
    n = len(matrix_a[0])  # кол-во столбцов в A = строк в B
    p = len(matrix_b[0])  # кол-во столбцов в B

    matrix_c = [[0 for _ in range(p)] for _ in range(m)]

    for i in range(m):
        for j in range(p):
            for k in range(n):
                matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return matrix_c
    pass


def functions(a_1, a_2):
    """
    Задание 2. На вход поступает две строки, содержащие коэффициенты двух функций.
    Необходимо найти точки экстремума функции и определить, есть ли у функций общие решения.
    Вернуть нужно координаты найденных решения списком, если они есть. None, если их бесконечно много.
    """
    a1 = list(map(float, a_1.split()))
    a2 = list(map(float, a_2.split()))

    # проверка на бесконечно много решений
    if a1 == a2:
        return None

    def F(x):
        return a1[0] * x ** 2 + a1[1] * x + a1[2]

    def P(x):
        return a2[0] * x ** 2 + a2[1] * x + a2[2]

    # точки экстремума
    res_F = minimize_scalar(F)
    res_P = minimize_scalar(P)
    extremum_F = res_F.x
    extremum_P = res_P.x

    extremum_points = []
    if not np.isnan(extremum_F):
        extremum_points.append((int(extremum_F), int(F(extremum_F))))
    if not np.isnan(extremum_P):
        extremum_points.append((int(extremum_P), int(P(extremum_P))))
    print("точки экстремума функции: ", extremum_points)

    roots = np.roots([a1[0] - a2[0], a1[1] - a2[1], a1[2] - a2[2]])
    real_roots = [r for r in roots if np.isreal(r)]

    solutions = []
    for r in real_roots:
        solutions.append((int(r.real), int(np.polyval(a1, r.real))))

    unique_solutions = sorted(set(solutions))
    return unique_solutions
    pass


def skew(x):
    """
    Задание 3. Функция для расчета коэффициента асимметрии.
    Необходимо вернуть значение коэффициента асимметрии, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    m3 = np.sum((x - mean) ** 3) / n
    skewness = m3 / std ** 3
    return round(skewness, 2)
    pass


def kurtosis(x):
    """
    Задание 3. Функция для расчета коэффициента эксцесса.
    Необходимо вернуть значение коэффициента эксцесса, округленное до 2 знаков после запятой.
    """
    n = len(x)
    mean = np.mean(x)
    std = np.std(x)
    m4 = np.sum((x - mean) ** 4) / n
    excess_kurtosis = m4 / std ** 4 - 3
    return round(excess_kurtosis, 2)
    pass
