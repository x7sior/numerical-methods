import numpy as np


def _sgn(numb):
    if numb < 0:
        return -1
    else:
        return 1


def _check_convergence(first, second, rtol: float = 1.e-6):
    return max(abs(first - second)) <= rtol * np.max([np.max([abs(first)]), np.max(np.abs([second]))])  # проверка сходимости


def _extend(matrix: np.ndarray, lower_bound, upper_bound):
    return matrix * (upper_bound - lower_bound) + lower_bound  # расширить область определения


def _normalize(vector):
    return vector / np.linalg.norm(vector)  # нормирование


def power_method(matrix, rtol: float = 1.e-6, delta: float = 1.e-8):
    n = len(matrix)
    y = np.random.rand(n)
    z = _normalize(y)
    y = matrix @ z
    lambda_new = np.zeros(n)

    for i in range(n):
        if abs(z[i]) > delta:
            lambda_new[i] = y[i] / z[i]

    z = _normalize(y)

    while True:
        y = matrix @ z
        lambda_new, lambda_old = np.zeros(n), lambda_new

        for i in range(n):
            if abs(z[i]) > delta:
                lambda_new[i] = y[i] / z[i]

        z = _normalize(y)

        if _check_convergence(lambda_new, lambda_old, rtol):
            return np.mean(lambda_new[lambda_new != 0]), z


def inverse_power_method(matrix, start_sigma, max_value, rtol: float = 1.e-6, delta: float = 1.e-8):
    n = len(matrix)
    sigma = np.array([start_sigma])
    y = np.random.rand(n)
    z, z_old = _normalize(y), _normalize(y)
    k = 0

    while True:
        try:
            y = np.linalg.solve(matrix - sigma * np.eye(n), z)
        except:
            return max_value, None  # вырожденная матрица
        mu = np.zeros(n)

        for i in range(n):
            if abs(y[i]) > delta:
                mu[i] = z[i] / y[i]
                
        z_old, z = z, _normalize(y)
        sigma_old, sigma = sigma, sigma + np.mean(mu[mu != 0])

        if _check_convergence(z_old, z, rtol) and _check_convergence(sigma, sigma_old, rtol):
            return sigma[0], z
        elif k == 100:
            return max_value, 3  # зацикливается
        
        k += 1


def find_by_inverse_power_method(matrix, bounds, rtol: float = 1.e-6):
    n = len(matrix)
    dig1, vec1 = power_method(matrix, rtol)
    dig, vec = [dig1], [vec1]
    lower_bound, upper_bound = bounds

    for i in range(1000):
        inp = (upper_bound - lower_bound) * i / 1000 + lower_bound
        dig1, vec1 = inverse_power_method(matrix, inp, dig[0], rtol)
        skip_flag = False

        for i in range(len(dig)):
            if abs(dig[i] - dig1) < 1.e-2:
                skip_flag = True

        if not skip_flag:
            dig += [dig1]
            vec += [vec1]

        if len(dig) == n:
            return dig, np.transpose(vec)
        
    k = 0
    while len(dig) != len(matrix):
        dig1, vec1 = inverse_power_method(matrix, _extend(np.random.rand(), lower_bound, upper_bound), dig[0], rtol)

        skip_flag = False
        for i in range(len(dig)):
            if abs(dig[i] - dig1) < 1.e-2:
                skip_flag = True
        
        if not skip_flag:
            dig += [dig1]
            vec += [vec1]

        if k == 10000:
            raise TimeoutError("Количество попыток подбора сдвига в обратном степенном методе превысило 10000, неудачная матрица")
        k += 1

    return dig, np.transpose(vec)


def _to_hessenberg(matrix, eps: float = 1.e-8):
    n = len(matrix)

    for i in range(n - 2):
        if np.linalg.norm(matrix[i + 2:, i]) < eps:
            continue

        s = _sgn(matrix[i + 1, i]) * np.linalg.norm(matrix[i + 1:, i])
        mu = 1 / np.sqrt(2 * s * (s - matrix[i + 1, i]))

        v = np.zeros((n, 1))
        v[i + 1] = matrix[i + 1, i] - s

        for j in range(i + 2, n):
            v[j] = matrix[j, i]
        
        v *= mu
        h = np.eye(n) - 2 * v @ np.transpose(v)
        matrix = h @ matrix @ h

    return matrix


def _qr_decomposition(matrix):
    n = len(matrix)
    q = np.eye(n)

    for i in range(n - 1):
        s = _sgn(-matrix[i, i]) * np.linalg.norm(matrix[i:, i])
        if np.sqrt(2 * s * (s - matrix[i, i])) < 1.e-24:
            continue

        mu = 1 / np.sqrt(2 * s * (s - matrix[i, i]))

        v = np.zeros((n, 1))
        v[i] = matrix[i, i] - s
        for j in range(i + 1, n):
            v[j] = matrix[j, i]
        v *= mu

        h = np.eye(n) - 2 * v @ np.transpose(v)
        matrix = h @ matrix
        q @= h

    return q, matrix


def qr_algorithm(matrix, rtol: float = 1.e-6, eps: float = 1.e-8):
    n = len(matrix)
    lamb = np.zeros(n)
    matrix = _to_hessenberg(matrix, eps)
    sigma = 0
    
    k = 0
    while True:
        if k == 100000:
            raise TimeoutError("Количество итераций QR-алгоритма превысило 100000")

        q, r = _qr_decomposition(matrix)
        sigma += matrix[-1, -1]
        matrix, old_matrix = r @ q - matrix[-1, -1] * np.eye(n), matrix

        if ((abs(matrix[-1, -2]) < eps and (abs(matrix[-1, -1] - old_matrix[-1, -1]) < rtol * abs(old_matrix[-1, -1]) or abs(matrix[-1, -1] - old_matrix[-1, -1]) < eps)) or abs(matrix[-1, -1]) < eps):
            n -= 1
            lamb[n] = sigma + matrix[-1, -1]
            matrix, old_matrix = (matrix + sigma * np.eye(n + 1))[:-1, :-1], (old_matrix + sigma * np.eye(n + 1))[:-1, :-1]
            sigma = 0
            if n == 1:
                lamb[0] = matrix[0, 0] + lamb[0]
                return lamb
        k += 1


def generate_matrix(n: int, bounds: tuple):
    lower_bound, upper_bound = bounds
    correct = False
    while not correct:  # Чтобы обратному степенному методу было проще подбирать сдвиги
        correct = True
        eigenvalues  = np.diag(_extend(np.random.rand(n), lower_bound, upper_bound))
        check = np.sort(eigenvalues [eigenvalues  != 0])

        for i in range(n - 1):
            if check[i + 1] - check[i] < 1:
                correct = False

    c = np.random.random((n, n))
    return np.linalg.inv(c) @ eigenvalues  @ c, eigenvalues 
    

