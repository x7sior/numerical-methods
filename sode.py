from typing import Callable

import numpy as np
import matplotlib.pyplot as plt


class SODE_Solver:  # System of Ordinary Differential Equations
    def __init__(self, func: Callable[[np.ndarray, np.ndarray], np.ndarray], x_start, y_start, x_end, order, total_err=1e-4, local_err=1e-5):
        self.func = func
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = np.array(y_start)

        self.order = order
        self.total_err = total_err
        self.local_err = local_err

        self.h = None
        self.log = None
        self.count_rhs_evals = False

    def _reset_state(self):
        self.log = {
            'nodes': [],
            'values': [],
            'h': [],
            'local_errors': [],
            'rhs_evals': 0  # right-hand side evaluations
        }
    
    def _runge_step(self, x_cur, y_cur, h):
        k = np.zeros((self.order, len(self.y_start)))
        k[0] = h * self.func(x_cur, y_cur)
        match self.order:
            case 2:
                k[1] = h * self.func(x_cur + h, y_cur + k[0])
                
                y_cur += (k[0] + k[1]) / 2

                if self.count_rhs_evals:
                    self.log['rhs_evals'] += 2

            case 3:
                k[1] = h * self.func(x_cur + h / 2, y_cur + k[0] / 2)
                k[2] = h * self.func(x_cur + h, y_cur - k[0] + 2 * k[1])

                y_cur += (k[0] + 4 * k[1] + k[2]) / 6

                if self.count_rhs_evals:
                    self.log['rhs_evals'] += 3

            case 4:
                k[1] = h * self.func(x_cur + h / 2, y_cur + k[0] / 2)
                k[2] = h * self.func(x_cur + h / 2, y_cur + k[1] / 2)
                k[3] = h * self.func(x_cur + h, y_cur + k[2])

                y_cur += (k[0] + 2 * k[1] + 2 * k[2] + k[3]) / 6

                if self.count_rhs_evals:
                    self.log['rhs_evals'] += 4

        x_cur += h
        return x_cur, y_cur

    def _set_start_h(self, total_err=None):
        if total_err is None:
            total_err = self.total_err
        delta = (1 / max(abs(self.x_start), abs(self.x_end)))**(self.order + 1) + (np.linalg.norm(self.func(self.x_start, self.y_start)))**(self.order + 1)
        return (total_err / delta) ** (1 / (self.order + 1))
 
    def const_step(self, h=None, do_log=False, total_err=None):
        self._reset_state()
        x_cur = self.x_start
        y_cur = self.y_start.copy()

        if h is None:
            h = self._set_start_h(total_err=total_err)

        while x_cur + h < self.x_end:
            x_cur, y_cur = self._runge_step(x_cur=x_cur, y_cur=y_cur, h=h)

            if do_log == True:
                self.log['nodes'].append(x_cur)
                self.log["values"].append(y_cur.copy())

        h = self.x_end - x_cur
        x_cur, y_cur = self._runge_step(x_cur=x_cur, y_cur=y_cur, h=h)

        if do_log == True:
            self.log['nodes'].append(x_cur)
            self.log['values'].append(y_cur.copy())

        self.h = h

        return y_cur

    def const_step_total_error(self, total_err=None):
        if total_err is None:
            total_err = self.total_err

        h = self._set_start_h(total_err=total_err)

        self.const_step(h=h, do_log=True)

        old_values = np.array(self.log["values"])

        while True:
            h /= 2
            self.const_step(h=h, do_log=True)

            new_values = self.log["values"]

            converged = True
            for i in range(len(old_values) - 1):
                if np.any(np.abs((old_values[i] - new_values[i * 2 + 1]) / (1 - 2**(-self.order))) > total_err):
                    converged = False
                    break

            last_err = (old_values[-1] - new_values[-1]) / (1 - 2**(-self.order))
            if np.any(np.abs(last_err) > total_err):
                converged = False
            
            if converged:
                self.h = h
                return new_values[-1] + last_err
            
            old_values = np.array(new_values)
    
    def _total_error_curve(self, total_err=None, h_ref=1e-4):
        self.const_step_total_error(total_err=total_err)
        nodes = np.array(self.log['nodes'])
        values = np.array(self.log['values'])
        total_error = []

        self._reset_state()
        x_cur = self.x_start
        y_cur = self.y_start.copy()

        for k in range(len(nodes)):
            h = h_ref
            while x_cur + h < nodes[k]:
                x_cur, y_cur = self._runge_step(x_cur=x_cur, y_cur=y_cur, h=h)

            h = self.x_end - x_cur
            x_cur, y_cur = self._runge_step(x_cur=x_cur, y_cur=y_cur, h=h)

            total_error.append(np.abs(y_cur - values[k]))

        return nodes, np.array(total_error)

    def auto_step(self, h_start=None, do_log=False, total_err=None, local_err=None):
        if local_err is None:
            local_err = self.local_err
        if h_start is None:
            h = self._set_start_h(total_err=total_err)
        else:
            h = h_start

        self._reset_state()
        x_cur = self.x_start
        y_cur = self.y_start.copy()

        step_accepted = True
        while x_cur < self.x_end:
            if step_accepted:
                x_old = x_cur  # x(i)
                y_old = y_cur.copy()  # y(i)
                x_cur, y_cur = self._runge_step(x_cur=x_cur, y_cur=y_cur, h=h)

            else:
                step_accepted = True

            y_h = y_cur.copy()  # y(i+1) с шагом h
            
            x_cur = x_old  # x(i)
            y_cur = y_old.copy()  # y(i)

            # вычисление y(i+1) с шагом h / 2 и x(i) + h
            h /= 2
            for _ in range(2):  
                x_cur, y_cur = self._runge_step(x_cur=x_cur, y_cur=y_cur, h=h)

            # оценка локальной погрешности для y(i+1) с шагом h
            error = np.linalg.norm((y_cur - y_h) / (1 - 2 ** -self.order))

            # не удовлетворяет предписанной точности
            if error > local_err * 2 ** self.order:
                x_cur = x_old  # x(i)
                y_cur = y_old.copy()  # y(i)
                step_accepted = False

            else:
                # не удовлетвоpяет точности, но может быть использовано приближение y(i+1) с шагом h / 2 (то есть текущие значения)
                if do_log == True:
                    self.log['nodes'].append(x_cur)
                    self.log['values'].append(y_cur.copy())
                    self.log['h'].append(h)
                    self.log['local_errors'].append(error)
                
                # удовлетворяет предписанной точности
                if error <= local_err:
                    h *= 2
                    y_cur = y_h.copy()  # y(i+1) с шагом h
                    
                    if do_log == True:
                        self.log['values'][-1] = y_cur.copy()

                    # оценка локальной погрешности существенно меньше заданной
                    if error < local_err / 2 ** (self.order + 1):
                        h *= 2

            if x_cur + h > self.x_end and step_accepted:
                h = self.x_end - x_cur

        return y_cur

    def _local_error_ratio(self, total_err=None, local_err=None, h_fer=1e-4):
        self.auto_step(do_log=True, total_err=total_err, local_err=local_err)
        nodes = np.array(self.log['nodes'])
        values = np.array(self.log['values'])
        h_values = np.array(self.log["h"])
        local_errors = np.array(self.log['local_errors'])
        local_error_ratios = []

        self._reset_state()
        x_cur = self.x_start
        y_cur = self.y_start.copy()

        for k in range(len(nodes)):
            h = h_fer
            while x_cur + h < nodes[k]:
                x_cur, y_cur = self._runge_step(x_cur=x_cur, y_cur=y_cur, h=h)

            h = nodes[k] - x_cur
            x_cur, y_cur = self._runge_step(x_cur=x_cur, y_cur=y_cur, h=h)

            local_error_ratios.append(np.linalg.norm(abs(y_cur - values[k])) / local_errors[k])
            y_cur = np.array(values[k])
        return nodes, h_values, np.array(local_error_ratios)

    def _cost_vs_tol(self, negativ_order_range: range = range(3, 11)):
        self.count_rhs_evals = True
        rhs_component_evals = []
        negativ_order = []

        for i in negativ_order_range:
            self.auto_step(local_err=10 ** -i)
            rhs_component_evals.append(self.log["rhs_evals"])
            negativ_order.append(i)

        self.count_rhs_evals = False
        return negativ_order, rhs_component_evals

    def show_const_step(self):
        plt.figure()
        x_axis, y_axis = self._total_error_curve()
        for i in range(len(self.y_start)):
            plt.plot(x_axis, y_axis[:, i], label=f'y{i+1}')
        plt.title(f'График зависимости истинной полной погрешности от значения x,\n при длине шага: {self.h} и порядке: {self.order}')
        plt.grid(True)
        plt.legend()
        plt.show()

    def show_auto_step(self):
        plt.figure()
        x_axis, h_values, local_error_ratios = self._local_error_ratio()
        plt.plot(x_axis, h_values)
        plt.title(f'График зависимости величины шага интегрирования от x,\n при порядке: {self.order}')
        plt.grid(True)

        plt.figure()
        plt.plot(x_axis, local_error_ratios * 100)
        plt.title(f'График зависимости отношения истинной локальной погрешности\n к полученной оценке локальной погрешности от x, при порядке: {self.order}, %')
        plt.grid(True)

        plt.figure()
        x_axis, y_axis = self._cost_vs_tol()
        plt.plot(x_axis, y_axis)
        plt.title(f'График зависимости количества вычислений правой части системы\n от заданной точности 10^(-x), при порядке: {self.order}')
        plt.grid(True)


