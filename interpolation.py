from typing import Callable

import matplotlib.pyplot as plt
import numpy as np


def create_table_uniform(
        func: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[float, float],
        n_nodes: int
) -> np.ndarray:
    """Строит таблицу равноотстоящих узлов и значений функции на отрезке bounds."""
    lower_bound, upper_bound = bounds
    nodes = np.linspace(lower_bound, upper_bound, n_nodes)
    return np.column_stack([nodes, func(nodes)])


def create_table_chebyshev(
        func: Callable[[np.ndarray], np.ndarray],
        bounds: tuple[float, float],
        n_nodes: int
) -> np.ndarray:
    """Строит таблицу узлов Чебышёва и значений функции на отрезке bounds."""
    lower_bound, upper_bound = bounds
    indices = np.arange(n_nodes)
    nodes = 0.5 * ((upper_bound - lower_bound) * np.cos((2 * indices + 1) / (2 * n_nodes) * np.pi) + (upper_bound + lower_bound))
    return np.column_stack([nodes, func(nodes)])


def create_lagrange_polynomial(table: np.ndarray) -> np.ndarray:
    n_nodes = len(table)
    lagrange_polynomial = np.array([0])
    for k in range(n_nodes):
        lagrange_multiplier = np.array([1])
        for i in range(n_nodes):
            if i != k:
                lagrange_multiplier = np.polymul(lagrange_multiplier, np.array([1, -table[i, 0]]))  # Умножение на (x - xi)
                lagrange_multiplier = lagrange_multiplier / (table[k, 0] - table[i, 0])  # Деление на (xk - xi)
        lagrange_polynomial = np.polyadd(lagrange_polynomial, lagrange_multiplier * table[k, 1])
    return lagrange_polynomial


def create_newton_polynomial(table: np.ndarray) -> np.ndarray:
    n_nodes = len(table)
    diff_table = np.zeros((n_nodes, n_nodes))
    diff_table[:, 0] = table[:, 1]  # Таблица значений и разделённых разностей

    omega = np.array([1.0])
    newton_polynomial = table[0, 1]
    for k in range(1, n_nodes):
        for i in range(n_nodes - k):  # Вычисление разделённых разностей k-ого порядка
            diff_table[i, k] = (diff_table[i + 1, k - 1] - diff_table[i, k - 1]) / (table[i + k, 0] - table[i, 0])
        omega = np.polymul(omega, np.array([1, -table[k - 1, 0]]))  # Омега порядка k
        newton_polynomial = np.polyadd(newton_polynomial, omega * diff_table[0, k])  # Полином Ньютона порядка k
    return newton_polynomial


def create_s10_spline(table: np.ndarray):
    spline_table = np.zeros((len(table) - 1, 2))
    for i in range(len(table) - 1):
        spline_table[i, 0], spline_table[i, 1] = np.linalg.solve(np.array([[table[i, 0], 1], [table[i + 1, 0], 1]]), np.array([table[i, 1], table[i + 1, 1]]))
    return spline_table


def create_s21_spline(table: np.ndarray, d: float):
    n = len(table)
    x_table = np.zeros(((n - 1) * 3, (n - 1) * 3))
    y_table = np.zeros(((n - 1) * 3))

    for i in range(n - 1):
        x_table[i * 3, i * 3] = table[i, 0]**2
        x_table[i * 3, i * 3 + 1] = table[i, 0]
        x_table[i * 3, i * 3 + 2] = 1

        x_table[i * 3 + 1, i * 3] = table[i + 1, 0]**2
        x_table[i * 3 + 1, i * 3 + 1] = table[i + 1, 0]
        x_table[i * 3 + 1, i * 3 + 2] = 1

        x_table[i * 3 + 2, i * 3] = 2 * table[i + 1, 0]
        x_table[i * 3 + 2, i * 3 + 1] = 1

        if i != n - 2:
            x_table[i * 3 + 2, i * 3 + 3] = -2 * table[i + 1, 0]
        if i != n - 2:
            x_table[i * 3 + 2, i * 3 + 4] = -1

        y_table[i * 3] = table[i, 1]
        y_table[i * 3 + 1] = table[i + 1, 1]
        y_table[i * 3 + 2] = 0 if i != n - 2 else d

    return np.linalg.solve(x_table, y_table).reshape((n - 1, 3))


def create_s32_spline(table: np.ndarray):
    n = len(table)
    step = (table[-1, 0] - table[0, 0]) / (n - 1)
    x_table = np.zeros(((n - 2), (n - 2)))
    y_table = np.zeros(n - 2)

    for i in range(n - 2):
        x_table[i, i] = 4 * step

        if i != n - 3:
            x_table[i, i + 1], x_table[i + 1, i] = step, step

        y_table[i] = (table[i + 2, 1] - 2 * table[i + 1, 1] + table[i, 1]) * 6 / step

    second_der = np.concatenate((np.array([0]), np.linalg.solve(x_table, y_table), np.array([0])))

    first_der = np.zeros(n - 1)
    for i in range(n - 1):
        first_der[i] = (table[i + 1, 1] - table[i, 1]) / step - second_der[i + 1] * step / 6 - second_der[i] * step / 3

    spline_table = np.zeros((n - 1, 4))
    for i in range(n - 1):
        x_shift = np.array([1, -table[i, 0]])  # (x - xi)
        x_shift2 = np.polymul(x_shift, x_shift)  # (x - xi)^2
        x_shift3 = np.polymul(x_shift2, x_shift)  # (x - xi)^3

        spline = np.polyadd(
            table[i, 1],
            first_der[i] * x_shift
        )
        spline = np.polyadd(
            spline,
            second_der[i] / 2 * x_shift2
        )
        spline = np.polyadd(
            spline,
            (second_der[i + 1] - second_der[i]) / (6 * step) * x_shift3
        )
        spline_table[i] = spline
    return spline_table


def solve_by_spline(spline_table: np.ndarray, points_table: np.ndarray):
    lower_bound, upper_bound = points_table[0], points_table[-1]
    step = (upper_bound - lower_bound) / len(spline_table)
    solves = np.zeros(len(points_table))

    for i in range(len(points_table)):
        j = int((points_table[i] - lower_bound) // step)
        if points_table[i] == upper_bound:
            j -= 1
            
        solves[i] = np.polyval(spline_table[j], points_table[i])
    return solves


def _find_interpolating_error(interpol_poly, table: np.ndarray):
    return abs(np.polyval(interpol_poly, table[:, 0]) - table[:, 1])


def _find_spline_error(spline, table: np.ndarray):
    return abs(np.array(solve_by_spline(spline, table[:, 0])) - table[:, 1])


def _find_max_interpolating_error(interpol_poly, table: np.ndarray):
    return max(_find_interpolating_error(interpol_poly, table))


def _find_max_spline_error(spline, table: np.ndarray):
    return max(_find_spline_error(spline, table))


def compare_lagrange_and_newton(
        func: Callable[[np.ndarray], np.ndarray],
        nm_pairs: tuple,
        bounds: tuple
    ):
    lagrange_table = np.zeros((len(nm_pairs), 4))
    newton_table = np.zeros((len(nm_pairs), 4))
    polynomial_table = [np.array([])] * 4

    plt.figure(figsize=(16, 8))
    f_table = create_table_uniform(func, bounds, 199)
    x_axis, f_values = f_table[:, 0], f_table[:, 1]

    for i in range(len(nm_pairs)):
        n_nodes, n_check_nodes = nm_pairs[i]

        # Заполнение таблиц
        lagrange_table[i, 0], newton_table[i, 0] = n_nodes, n_nodes  # Количество узлов
        lagrange_table[i, 1], newton_table[i, 1] = n_check_nodes, n_check_nodes  # Количество проверочных узлов

        # Создание таблиц узлов и значений по равноотстоящим узлам и узлам Чебышева
        table_un, table_cheb = create_table_uniform(func, bounds, n_nodes), create_table_chebyshev(func, bounds, n_nodes)
        table_un_check, table_cheb_check = create_table_uniform(func, bounds, n_check_nodes), create_table_chebyshev(func, bounds, n_check_nodes)

        # Вычисление интерполяционных полиномов
        polynomial_table[0] = create_lagrange_polynomial(table_un)
        polynomial_table[1] = create_lagrange_polynomial(table_cheb)
        polynomial_table[2] = create_newton_polynomial(table_un)
        polynomial_table[3] = create_newton_polynomial(table_cheb)

        # Максимальное отклонение по равноотстоящим узлам
        lagrange_table[i][2] = _find_max_interpolating_error(polynomial_table[0], table_un_check)
        newton_table[i][2] = _find_max_interpolating_error(polynomial_table[2], table_un_check)
        # Максимальное отклонение по узлам Чебышева
        lagrange_table[i][3] = _find_max_interpolating_error(polynomial_table[1], table_cheb_check)
        newton_table[i][3] = _find_max_interpolating_error(polynomial_table[3], table_cheb_check)

        # Создание графиков
        plt.subplot(len(nm_pairs), 2, i * 2 + 1)
        plt.plot(x_axis, f_values, label=f'Изначальная функция')
        plt.plot(x_axis, np.polyval(polynomial_table[0], x_axis), label=f'по равноотстоящим узлам')
        plt.plot(x_axis, np.polyval(polynomial_table[1], x_axis), label=f'по узлам Чебышева')
        plt.title(f'Полином Лагранжа порядка: {len(polynomial_table[0]) - 1}')
        plt.grid(True)
        plt.legend()

        plt.subplot(len(nm_pairs), 2, i * 2 + 2)
        plt.plot(x_axis, f_values, label=f'Изначальная функция')
        plt.plot(x_axis, np.polyval(polynomial_table[2], x_axis), label=f'по равноотстоящим узлам')
        plt.plot(x_axis, np.polyval(polynomial_table[3], x_axis), label=f'по узлам Чебышева')
        plt.title(f'Полином Ньютона порядка: {len(polynomial_table[0]) - 1}')
        plt.grid(True)
        plt.legend()

    # Вывод результатов
    def _print_table(table: np.ndarray, n: int):
        ltext = [["Количество", "Количество", "Максимальное", "Максимальное"],
                 ["узлов", "проверочных", "отклонение", "отклонение"],
                 ["(n)", "узлов (m)", "(Rl)", "(RLopt)"]]
        for i in range(3):
            print(end="\n |")
            for j in range(4):
                print(f"{ltext[i][j]:^22}", end="|")

        ltable = table.tolist()
        for i in range(n):
            ltable[i][0], ltable[i][1] = int(ltable[i][0]), int(ltable[i][1])
        for i in range(n):
            print(end="\n |")
            for j in range(4):
                print(f"{ltable[i][j]:^22}", end="|")
        print("\n")

    print("\n Поведение интерполяционного полинома Лагранжа при увеличении количества узлов интерполирования.")
    _print_table(lagrange_table, len(nm_pairs))
    print("\n Поведение интерполяционного полинома Ньютона при увеличении количества узлов интерполирования.")
    _print_table(newton_table, len(nm_pairs))
    plt.tight_layout()
    plt.show()


def compare_splines_and_newton(
        func: Callable[[np.ndarray], np.ndarray],
        nm_pairs: tuple,
        bounds: tuple,
        d=float
    ):
    table_s10 = np.zeros((len(nm_pairs), 3))
    table_s21 = np.zeros((len(nm_pairs), 3))
    table_s32 = np.zeros((len(nm_pairs), 3))
    polynomial_table = [np.array([])] * 4

    plt.figure(figsize=(17, 8))
    f_table = create_table_uniform(func, bounds, 200)
    x_axis = f_table[:, 0]
    for i in range(len(nm_pairs)):
        n_nodes, n_check_nodes = nm_pairs[i]

        # Заполнение таблиц
        table_s10[i, 0], table_s10[i, 1] = n_nodes, n_check_nodes
        table_s21[i, 0], table_s21[i, 1] = n_nodes, n_check_nodes
        table_s32[i, 0], table_s32[i, 1] = n_nodes, n_check_nodes

        table, table_check = create_table_uniform(func, bounds, n_nodes), create_table_uniform(func, bounds, n_check_nodes)

        # Вычисление интерполяционных полиномов
        polynomial_table[0] = create_s10_spline(table)
        polynomial_table[1] = create_s21_spline(table, d)
        polynomial_table[2] = create_s32_spline(table)
        polynomial_table[3] = create_newton_polynomial(table)

        # Максимальное отклонение по равноотстоящим узлам
        table_s10[i, 2] = _find_max_spline_error(polynomial_table[0], table_check)
        table_s21[i, 2] = _find_max_spline_error(polynomial_table[1], table_check)
        table_s32[i, 2] = _find_max_spline_error(polynomial_table[2], table_check)

        # Графики интерполяционных сплайнов
        plt.subplot(len(nm_pairs), 2, i * 2 + 1)
        plt.plot(x_axis, solve_by_spline(polynomial_table[0], x_axis), label=f'Линейный сплайн S1,0')
        plt.plot(x_axis, solve_by_spline(polynomial_table[1], x_axis), label=f'Квадратичный сплайн S2,1')
        plt.plot(x_axis, solve_by_spline(polynomial_table[2], x_axis), label=f'Кубический сплайн S3,2')
        plt.title(f'Интерполяционные сплайны по {n_nodes} узлам')
        plt.grid(True)
        plt.legend()

        # Графики распределения абсолютной погрешности
        plt.subplot(len(nm_pairs), 2, i * 2 + 2)
        plt.plot(x_axis, _find_spline_error(polynomial_table[2], f_table), label=f'Кубический сплайн S3,2')
        plt.plot(x_axis, _find_interpolating_error(polynomial_table[3], f_table), label=f'Полином Ньютона')
        plt.title(f'Распределения абсолютной погрешности на интервале интерполирования по {n_check_nodes} узлам')
        plt.grid(True)
        plt.legend()

    # Вывод результатов
    def print_table(table: np.ndarray, n: int):
        ltext = [["Количество", "Количество", "Максимальное"],
                 ["узлов", "проверочных", "отклонение"],
                 ["(n)", "узлов (m)", "(RSnm,p)"]]
        for i in range(3):
            print(end="\n |")
            for j in range(3):
                print(f"{ltext[i][j]:^22}", end="|")

        ltable = table.tolist()
        for i in range(n):
            ltable[i][0], ltable[i][1] = int(ltable[i][0]), int(ltable[i][1])
        for i in range(n):
            print(end="\n |")
            for j in range(3):
                print(f"{ltable[i][j]:^22}", end="|")
        print("\n")

    print("\n Поведение линейных сплайнов S1,0 при увеличении количества узлов интерполирования.")
    print_table(table_s10, len(nm_pairs))
    print("\n Поведение квадратичных сплайнов S2,1 при увеличении количества узлов интерполирования.")
    print_table(table_s21, len(nm_pairs))
    print("\n Поведение кубический сплайн S3,2 при увеличении количества узлов интерполирования.")
    print_table(table_s32, len(nm_pairs))
    plt.tight_layout()
    plt.show()
