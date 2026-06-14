from typing import Literal, Callable

import numpy as np
import matplotlib.pyplot as plt


class Integration:
    def __init__(
            self,
            func: Callable[[float], float] = None,
            find_mu: Callable[[tuple[float, float], float, float, int], np.ndarray] = None,
            bounds: tuple[float, float] = (2.1, 3.3),
            eps: float = 1e-6
    ):
        if func:
            self.func = func
        if find_mu:
            self.find_mu = find_mu

        self.bounds = bounds
        self.eps = eps

    @staticmethod
    def func(x: float) -> float:
        return 4.5 * np.cos(7 * x) * np.exp(-2 * x / 3) + 1.4 * np.sin(1.5 * x) * np.exp(-x / 3) + 3

    @staticmethod
    def find_mu(
        bounds: tuple[float, float],
        z1: float,
        z2: float,
        n: int
    ) -> np.ndarray:
        lower_bound = bounds[0]
        alpha = 2 / 5
        terms = np.arange(n) + 1 - alpha
        return ((z2 - lower_bound)**terms - (z1 - lower_bound)**terms) / terms

    @staticmethod
    def _sgn(num: float):
        if num >= 0:
            return 1
        else:
            return num / abs(num)

    def midpoint_rectangles(self, n: int) -> float:
        lower_bound, upper_bound = self.bounds
        h = (upper_bound - lower_bound) / n
        return h * np.sum(
            self.func(np.linspace(lower_bound + h / 2, upper_bound - h / 2, n))
        )

    def left_rectangles(self, n: int) -> float:
        lower_bound, upper_bound = self.bounds
        h = (upper_bound - lower_bound) / n
        return h * np.sum(
            self.func(np.linspace(lower_bound, upper_bound - h, n))
        )

    def trapezoidal_rule(self, n: int) -> float:
        lower_bound, upper_bound = self.bounds
        h = (upper_bound - lower_bound) / n
        return h * (
            (self.func(lower_bound) + self.func(upper_bound)) / 2
            + np.sum(
                self.func(np.linspace(lower_bound + h, upper_bound - h, n - 1))
            )
        )

    def simpsons_rule(self, n: int) -> float:
        return (self.trapezoidal_rule(n) + 2 * self.midpoint_rectangles(n)) / 3

    def _roots3(self, polynomial: np.ndarray) -> np.ndarray:
        # Формула кардано
        a, b, c, d = polynomial
        # после замены на y = x + b / 3a
        q = b ** 3 / (27 * a**3) - (b * c) / (6 * a**2) + d / (2 * a)
        p = (3 * a * c - b**2) / (9 * a**2)

        r = self._sgn(q) * abs(abs(p)**(0.5))

        cos_phi = q / r**3
        cos_phi = max(-1.0, min(1.0, cos_phi))  # защита от ошибок из-за погрешности округления

        phi = np.arccos(cos_phi)
        coeff = 2 * r

        y = np.array([
            -coeff * np.cos(phi / 3),
            coeff * np.cos((np.pi - phi) / 3),
            coeff * np.cos((np.pi + phi) / 3)
        ])

        return y - b / (3 * a)  # обратная замена x = y - b / 3a

    def newton_cotes(
        self,
        z1: float,
        z2: float,
        n: int
    ) -> float:
        lower_bound = self.bounds[0]
        h = (z2 - z1) / (n - 1)

        x = z1 - lower_bound + np.arange(n) * h
        matrix_x = np.vander(x, increasing=True).T
        vec_a = np.linalg.solve(matrix_x, self.find_mu(self.bounds, z1, z2, n))

        return vec_a @ self.func(matrix_x[1] + lower_bound)

    def gaussian(
        self,
        z1: float,
        z2: float,
        n: int
    ) -> float:
        lower_bound = self.bounds[0]
        mu = self.find_mu(self.bounds, z1, z2, n * 2)

        indices = np.arange(n)
        matrix_mu = mu[indices[:, None] + indices]
        coef_a = np.linalg.solve(matrix_mu, -mu[n:])

        x_g = self._roots3(np.insert(coef_a[::-1], 0, 1))

        mat_x = np.vander(x_g, increasing=True).T
        vec_a = np.linalg.solve(mat_x, mu[:n])

        return vec_a @ self.func(x_g + lower_bound)

    # Составная квадратурную формула
    def skf(
            self,
            method: Literal['newton_cotes', 'gaussian'],
            n: int
    ) -> float:
        match method:
            case 'newton_cotes':
                skf_method = self.newton_cotes
            case 'gaussian':
                skf_method = self.gaussian

        lower_bound, upper_bound = self.bounds
        h = (upper_bound - lower_bound) / n

        return sum(
            skf_method(z1=lower_bound + i * h, z2=lower_bound + (i + 1) * h, n=3)
            for i in range(n)
        )

    @staticmethod
    def aitken(approximations: list, ratio: int) -> float:
        a0, a1, a2 = approximations[-3:]
        return -np.log(abs((a2 - a1) / (a1 - a0))) / np.log(ratio)

    @staticmethod
    def richardson(
        approximations: list,
        leading_order: float,  # m
        h: float,
        ratio: int = 2  # L
    ) -> tuple[np.ndarray, float]:
        extra_terms = len(approximations) - 1  # r

        i = np.arange(extra_terms + 1)
        h_vec = (h / ratio**i)[:, None]  # (h, h/2, h/4, ...)^T

        j = np.arange(extra_terms)
        h_matrix = h_vec**(leading_order + j)  # (h^m, h^(m+1), h^(m+2), ...)

        matrix = np.hstack([-h_matrix, np.ones((extra_terms + 1, 1))])  # (-h^m, -h^(m+1), ..., -h^(m+r-1), 1)

        solution = np.linalg.solve(matrix, approximations)  # Решение системы: (-h^m, -h^(m+1), ..., -h^(m+r-1), 1) @ (C_m, C_(m+1), ..., C_(m+r-1), J(f))^T = S_h
        richardson_estimate = solution[extra_terms]  # J(f)

        return abs(richardson_estimate - approximations), richardson_estimate  # (R_h, J(f))


    def skf_error_estimation(
        self,
        method: Literal['newton_cotes', 'gaussian'],
        n_start: int = 1,
        ratio: int = 2
    ) -> dict:
        match method:
            case 'newton_cotes':
                leading_order_initial = 3
            case 'gaussian':
                leading_order_initial = 6

        skf = self.skf

        lower_bound, upper_bound = self.bounds
        h = (upper_bound - lower_bound) / n_start
        leading_order = leading_order_initial

        approximations = [skf(method=method, n=n_start * ratio**k) for k in range(2)]  # S_h1 и S_h2
        errors, richardson_estimate = self.richardson(approximations=approximations, leading_order=leading_order, h=h, ratio=ratio)

        if errors[0] < self.eps:
            return {'value': richardson_estimate, 'n': n_start}

        if errors[-1] < self.eps:
            return {'value': richardson_estimate, 'n': n_start * ratio}

        k = 2
        while True:
            approximations.append(skf(method=method, n=n_start * ratio**k))
            errors, richardson_estimate = self.richardson(approximations=approximations, leading_order=leading_order, h=h, ratio=ratio)
            leading_order = self.aitken(approximations=approximations, ratio=ratio)

            if errors[-1] <= self.eps:
                return {'value': richardson_estimate, 'n': n_start * ratio**k}

            k += 1

    def find_optimal_n(
        self,
        method: Literal['newton_cotes', 'gaussian'],
        ratio: int = 2
    ) -> int:
        skf = self.skf

        approximations = [skf(method=method, n=ratio**i) for i in range(3)]
        leading_order = self.aitken(approximations=approximations, ratio=ratio)

        n_opt = int(
            (
                abs(approximations[1] - approximations[0]) / (self.eps * (1 - ratio**(-leading_order)))
            )**(1 / leading_order)
        )

        return n_opt

    def integration_test_unweighted(self, true_value: float):
        x_axis = list(range(1, 10))
        plt.figure()
        plt.plot(x_axis, [abs(self.midpoint_rectangles(i) - true_value) for i in x_axis], label="Формула средних прямоугольников")
        plt.plot(x_axis, [abs(self.left_rectangles(i) - true_value) for i in x_axis], label="Формула левых прямоугольников")
        plt.plot(x_axis, [abs(self.trapezoidal_rule(i) - true_value) for i in x_axis], label="Формула трапеции")
        plt.plot(x_axis, [abs(self.simpsons_rule(i) - true_value) for i in x_axis], label="Формула Симпсона")
        plt.title('График зависимости абсолютной погрешности\n от количества разбиений интервала интегрирования')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()

        plt.show()

    def integration_test_weighted(self, true_value: float):
        x_axis = list(range(1, 10))
        plt.figure()
        plt.plot(x_axis, [abs(self.skf(method='newton_cotes', n=i) - true_value) for i in x_axis], label="Формула Ньютона-Котса")
        plt.plot(x_axis, [abs(self.skf(method='gaussian', n=i) - true_value) for i in x_axis], label="Формула Гаусса")
        plt.title('График зависимости абсолютной погрешности\n от количества разбиений интервала интегрирования')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()

        print(f"Шаг разбиения интервала интегрирования для достижения заданной точности методом Ньютона-Котса: 1 / {self.skf_error_estimation(method='newton_cotes')['n']}")
        n_opt = self.find_optimal_n(method='gaussian')
        print(f"Шаг разбиения интервала интегрирования для достижения заданной точности методом Ньютона-Котса, начиная с оптимального шага: 1 / {self.skf_error_estimation(method='newton_cotes', n_start=n_opt)['n']}")
        print(f"Шаг разбиения интервала интегрирования для достижения заданной точности методом Гаусса: 1 / {self.skf_error_estimation(method='gaussian')['n']}")
        n_opt = self.find_optimal_n(method='gaussian')
        print(f"Шаг разбиения интервала интегрирования для достижения заданной точности методом Гаусса, начиная с оптимального шага: 1 / {self.skf_error_estimation(method='gaussian', n_start=n_opt)['n']}")
        plt.show()
