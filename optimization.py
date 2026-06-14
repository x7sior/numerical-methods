from typing import Callable

import numpy as np


def simplex_method(
        func: Callable[[np.ndarray], float],
        x_start: np.ndarray | list = None,
        size: int = None,
        alpha:float = 1.,
        gamma:float = 2.,
        beta:float = 0.5,
        delta:float = 0.5,
        eps:float = 1e-5,
        count_iters: bool = False
    ):
    if x_start is None:
        if size is None:
            simplex = [np.random.randn(1) * 0.01]
        else:
            simplex = [np.random.randn(size) * 0.01]

    else:
        simplex = [np.array(x_start, dtype=float)]

    for i in range(len(simplex[0])):
        y = simplex[0].copy()

        if y[i]:
            y[i] *= 1.05
        else:
            y[i] = 0.05

        simplex.append(y)
        
    simplex = np.array(simplex, dtype=float)

    iters = 0
    while True:
        iters += 1
        simplex = sorted(simplex, key=func)

        best = simplex[0]
        worst = simplex[-1]
        second_worst = simplex[-2]

        centroid = np.mean(simplex[:-1], axis=0)
        reflected = centroid + alpha * (centroid - worst)
        f_reflected = func(reflected)

        if func(best) <= f_reflected < func(second_worst):
            simplex[-1] = reflected

        elif f_reflected < func(best):
            expanded = centroid + gamma * (reflected - centroid)

            if func(expanded) < f_reflected:
                simplex[-1] = expanded
            else:
                simplex[-1] = reflected

        else:
            contracted = centroid + beta * (worst - centroid)
            if func(contracted) < func(worst):
                simplex[-1] = contracted

            else:
                for i in range(1, len(simplex)):
                    simplex[i] = best + delta * (simplex[i] - best)

        if np.max(np.abs(simplex[0] - simplex[1:])) < eps:
            if count_iters:
                return simplex[0], iters
            else:
                return simplex[0]


def golden_section_search(func, a, b, eps=1e-5):
    tau = 0.6180339887498949  # (sqrt(5) - 1) / 2

    while np.linalg.norm(b - a) > eps:
        c = b - (b - a) * tau
        d = a + (b - a) * tau

        if func(c) < func(d):
            b = d
        else:
            a = c

    return (a + b) / 2


class Gradient_Descent:
    def __init__(
            self,
            func: Callable[[np.ndarray], float],
            grad: Callable[[np.ndarray], np.ndarray] = None,
            x_start: np.ndarray | list = None,
            lr: float = 1e-3,
            size: int = None,
            eps_grad: float = 1.e-5,
            eps_func: float = 1.e-5,
            eps_x: float = 1.e-5
        ):
        self.func = func
        if grad:
            self.grad = grad

        if x_start is None:
            if size is None:
                self.x_start = np.random.randn(1) * 0.01
            else:
                self.x_start = np.random.randn(size) * 0.01
        else:
            self.x_start = np.array(x_start, dtype=float)

        self.lr = lr
        self.eps_grad = eps_grad
        self.eps_func = eps_func
        self.eps_x = eps_x
        self.iters = None
        self.x = self._run()
        
    def grad(self, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        func = self.func
        grad_value = []

        for i in range(len(x)):
            delta = np.zeros(len(x))
            delta[i] = eps
            grad_value.append((func(x + delta) - func(x - delta)) / (2 * eps))

        return np.array(grad_value)

    def _check_stopping_criterion(self, x: np.ndarray, new_x: np.ndarray, value: float, new_value: float, grad_value: np.ndarray):
        if self.eps_grad:
            if np.linalg.norm(grad_value) < self.eps_grad:
                return True
        if self.eps_func:
            if np.abs(new_value - value) < self.eps_func:
                return True
        if self.eps_x:
            if np.all(np.abs(new_x - x) < self.eps_x):
                return True
        return False

    def _run(self):
        func = self.func
        grad = self.grad        
        check_stopping_criterion = self._check_stopping_criterion


        x = self.x_start.copy()
        value = func(x)
        grad_value = grad(x)
        lr = self.lr

        iters = 1
        while True:
            new_x = x - lr * grad_value
            new_value = func(new_x)

            if check_stopping_criterion(x=x, new_x=new_x, value=value, new_value=new_value, grad_value=grad_value):
                self.iters = iters
                return new_x
            
            iters += 1
            x = new_x
            value = new_value
            grad_value = grad(x)


class Steepest_Descent(Gradient_Descent):
    def _run(self):
        func = self.func
        grad = self.grad        
        check_stopping_criterion = self._check_stopping_criterion

        x = self.x_start.copy()
        value = func(x)
        grad_value = grad(x)

        iters = 1
        while True:
            new_x = golden_section_search(func=func, a=x, b=x - grad_value)
            new_value = func(new_x)

            if check_stopping_criterion(x=x, new_x=new_x, value=value, new_value=new_value, grad_value=grad_value):
                self.iters = iters
                return new_x
            
            iters += 1
            x = new_x
            value = new_value
            grad_value = grad(x)


class Conjugate_Gradient(Gradient_Descent):
    def __init__(
            self,
            func: Callable[[np.ndarray], float],
            grad: Callable[[np.ndarray], np.ndarray] = None,
            x_start: np.ndarray | list = None,
            size: int = None,
            eps_grad: float = 1.e-5,
            eps_func: float = 1.e-5,
            eps_x: float = 1.e-5,
            eps_g: float = 1.e-5
        ):
        self.eps_g = eps_g
        super().__init__(func=func, grad=grad, x_start=x_start, size=size, eps_grad=eps_grad, eps_func=eps_func, eps_x=eps_x)

    def _run(self):
        func = self.func
        grad = self.grad
        check_stopping_criterion = self._check_stopping_criterion

        eps_g = self.eps_g
        x = self.x_start.copy()
        value = func(x)
        grad_value = grad(x)

        delta = -grad_value
        dimensions = len(x)
        iters = 1

        while True:
            new_x = golden_section_search(func=func, a=x, b=x + delta, eps=eps_g)
            old_grad_value = grad_value
            grad_value = grad(new_x)

            if (iters + 1) % dimensions == 0:
                delta = -grad_value
            else:
                beta = np.linalg.norm(grad_value)**2 / np.linalg.norm(old_grad_value)**2
                delta = -grad_value + beta * delta

            new_value = func(new_x)

            if check_stopping_criterion(x=x, new_x=new_x, value=value, new_value=new_value, grad_value=grad_value):
                self.iters = iters
                return new_x
            
            iters += 1
            value = new_value
            x = new_x.copy()
            grad_value = grad(x)
            

class Newtone_Method(Gradient_Descent):
    def __init__(
            self,
            func: Callable[[np.ndarray], float],
            grad: Callable[[np.ndarray], np.ndarray] = None,
            grad_2: Callable[[np.ndarray], np.ndarray] = None,
            x_start: np.ndarray | list = None,
            lr: float = 1.,
            size: int = None,
            eps_grad: float = 1.e-5,
            eps_func: float = 1.e-5,
            eps_x: float = 1.e-5,
            eps_g: float = 1.e-5
        ):
        self.grad_2 = grad_2
        self.eps_g = eps_g
        super().__init__(func=func, grad=grad, x_start=x_start, lr=lr, size=size, eps_grad=eps_grad, eps_func=eps_func, eps_x=eps_x)

    def _run(self):
        check_stopping_criterion = self._check_stopping_criterion

        func = self.func
        grad = self.grad
        grad_2 = self.grad_2


        x = self.x_start.copy()
        grad_value = grad(x)
        value = func(x)
        lr = self.lr
        eps_g = self.eps_g
        iters = 1

        while True:
            new_x = golden_section_search(func=func, a=x, b=lr * (x - np.linalg.inv(grad_2(x)) @ grad_value), eps=eps_g)
            new_value = func(new_x)

            if check_stopping_criterion(x=x, new_x=new_x, value=value, new_value=new_value, grad_value=grad_value):
                self.iters = iters
                return new_x
            
            iters += 1
            x = new_x
            value = new_value
            grad_value = grad(x)
