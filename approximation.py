from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


def _create_table(
        func: Callable[[np.ndarray], np.ndarray],
        n_nodes: int,
        bounds: tuple,
        values_koef=3
    ) -> np.ndarray:
    lower_bound, upper_bound = bounds
    table = np.zeros((2, n_nodes * values_koef))
    x = np.linspace(lower_bound, upper_bound, n_nodes)

    for i in range(n_nodes):
        for j in range(values_koef):
            table[0, i * values_koef + j] = x[i]
            table[1, i * values_koef + j] = func(x[i])

    return table


def norm_mnk(table, n):
    n += 1
    matrix = np.zeros((n, n))
    vec = np.zeros((n, 1))

    for i in range(n):
        for j in range(n):
            matrix[i, j] = sum(table[0] ** (i + j))

        vec[i] = sum(table[0]**i * table[1])

    return np.linalg.solve(matrix, vec)[::-1, 0]


def ort_mnk(table, n):
    n += 1
    q = [np.ndarray] * (n + 1)
    q[0] = np.array([1])
    q[1] = np.array([1, -sum(table[0]) / len(table[0])])
    polynomial = np.zeros(n)

    for i in range(1, n - 1):
        alpha = sum(table[0] * np.polyval(q[i], table[0]) ** 2) / sum(np.polyval(q[i], table[0])**2)
        beta = sum(table[0] * np.polyval(q[i], table[0]) * np.polyval(q[i - 1], table[0])) / sum(np.polyval(q[i - 1], table[0])**2)
        q[i + 1] = np.polyadd(np.polyadd(np.polymul(q[i], [1, 0]), -alpha * q[i]), -beta * q[i - 1])

    for i in range(n):
        polynomial = np.polyadd(polynomial, q[i] * sum(np.polyval(q[i], table[0]) * table[1]) / sum(np.polyval(q[i], table[0]) ** 2))

    return polynomial


def mnk_test(
        func: Callable[[np.ndarray], np.ndarray],
        bounds: tuple,
        n_nodes: int,
        values_koef: int = 3
    ):
    task_table = np.zeros((6, 3))
    polynomial_table = [np.ndarray] * 2

    plt.figure(figsize=(17, 8))
    x_axis = np.linspace(bounds[0], bounds[1], 200)

    table = _create_table(func, n_nodes, bounds, values_koef)
    for n in range(1, 7):
        task_table[n - 1, 0] = n
        polynomial_table[0] = ort_mnk(table, n)
        polynomial_table[1] = norm_mnk(table, n)
        task_table[n - 1, 1] = sum(table[1] - np.polyval(polynomial_table[0], table[0]) ** 2)
        task_table[n - 1, 2] = sum(table[1] - np.polyval(polynomial_table[1], table[0]) ** 2)

        plt.subplot(3, 2, n)
        plt.plot(x_axis, np.polyval(polynomial_table[0], x_axis), label=f'МНК (нормальные уравнения)')
        plt.plot(x_axis, np.polyval(polynomial_table[1], x_axis), label=f'МНК (ортогональные многочлены)')
        plt.scatter(table[0], table[1], label=f'Экспериментальные точки')
        plt.title(f'Результаты аппроксимации функции полиномами {n} степени')
        plt.grid(True)
        plt.legend()

    # Вывод результатов
    def print_table(table: np.ndarray, n: int):
        ltext = [["Степень", "Сумма квадратов", "Сумма квадратов"],
                 ["полинома", "ошибок для МНК", "ошибок для МНК"],
                 ["(n)", "(нормальные", "(ортогональные"],
                 ["", "уравнения)", "полиномы)"]]
        
        for i in range(len(ltext)):
            print(end="\n |")
            for j in range(len(ltext[0])):
                print(f"{ltext[i][j]:^22}", end="|")

        ltable = table.tolist()
        for i in range(n):
            ltable[i][0] = int(ltable[i][0])

        for i in range(n):
            print(end="\n |")
            for j in range(3):
                print(f"{ltable[i][j]:^22}", end="|")

        print("\n")

    print_table(task_table, 6)
    plt.tight_layout()
    plt.show()