"""Local numerical helpers used across the package."""

from __future__ import annotations

import math
from typing import Callable


def linspace(start: float, stop: float, count: int) -> list[float]:
    """Return `count` evenly spaced points."""
    if count <= 0:
        return []
    if count == 1:
        return [float(start)]
    step = (stop - start) / float(count - 1)
    return [start + step * float(index) for index in range(count)]


def simpson_integral(func: Callable[[float], float], a: float, b: float, n: int) -> float:
    """Integrate a scalar function with Simpson's rule."""
    if not math.isfinite(a) or not math.isfinite(b):
        return math.nan
    if b < a:
        return 0.0
    if abs(b - a) < 1.0e-14:
        return 0.0
    if n % 2 == 1:
        n += 1

    x_values = linspace(a, b, n + 1)
    y_values = [float(func(x_value)) for x_value in x_values]
    h = (b - a) / float(n)

    total = y_values[0] + y_values[-1]
    total += 4.0 * sum(y_values[1:-1:2])
    total += 2.0 * sum(y_values[2:-1:2])
    return h * total / 3.0


def simpson_weights(a: float, b: float, n: int) -> tuple[list[float], list[float]]:
    """Return grid points and Simpson weights."""
    if n % 2 == 1:
        n += 1
    x_values = linspace(a, b, n + 1)
    h = (b - a) / float(n)
    weights = [1.0] * (n + 1)
    for index in range(1, n):
        weights[index] = 4.0 if index % 2 == 1 else 2.0
    weights = [weight * h / 3.0 for weight in weights]
    return x_values, weights


def bisection_root(
    func: Callable[[float], float | None],
    left: float,
    right: float,
    tol: float = 1.0e-10,
    max_iter: int = 300,
) -> float | None:
    """Find a root in `[left, right]` using bisection."""
    f_left = func(left)
    f_right = func(right)
    if f_left is None or f_right is None:
        return None
    if f_left == 0.0:
        return left
    if f_right == 0.0:
        return right
    if f_left * f_right > 0.0:
        return None

    a = left
    b = right
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        f_mid = func(mid)
        if f_mid is None:
            return None
        if abs(f_mid) < tol or abs(b - a) < tol:
            return mid
        if f_left * f_mid < 0.0:
            b = mid
            f_right = f_mid
        else:
            a = mid
            f_left = f_mid
    return 0.5 * (a + b)


def golden_section_min(
    func: Callable[[float], float],
    left: float,
    right: float,
    tol: float = 1.0e-8,
    max_iter: int = 400,
) -> tuple[float, float]:
    """Minimize a scalar function on a closed interval."""
    golden = 0.5 * (math.sqrt(5.0) - 1.0)
    a = left
    b = right
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    f_c = func(c)
    f_d = func(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f_c < f_d:
            b = d
            d = c
            f_d = f_c
            c = b - golden * (b - a)
            f_c = func(c)
        else:
            a = c
            c = d
            f_c = f_d
            d = a + golden * (b - a)
            f_d = func(d)

    x_star = 0.5 * (a + b)
    return x_star, func(x_star)


def golden_section_min_safe(
    func: Callable[[float], float],
    left: float,
    right: float,
    tol: float = 1.0e-8,
    max_iter: int = 400,
) -> tuple[float, float]:
    """Golden-section minimization that tolerates `nan` regions."""
    golden = 0.5 * (math.sqrt(5.0) + 1.0)
    a = left
    b = right
    c = b - (b - a) / golden
    d = a + (b - a) / golden
    f_c = func(c)
    f_d = func(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if (not math.isfinite(f_c)) and (not math.isfinite(f_d)):
            c = b - (b - a) / golden
            d = a + (b - a) / golden
            f_c = func(c)
            f_d = func(d)
            continue

        if not math.isfinite(f_c):
            a = c
            c = d
            f_c = f_d
            d = a + (b - a) / golden
            f_d = func(d)
            continue

        if not math.isfinite(f_d):
            b = d
            d = c
            f_d = f_c
            c = b - (b - a) / golden
            f_c = func(c)
            continue

        if f_c < f_d:
            b = d
            d = c
            f_d = f_c
            c = b - (b - a) / golden
            f_c = func(c)
        else:
            a = c
            c = d
            f_c = f_d
            d = a + (b - a) / golden
            f_d = func(d)

    x_star = 0.5 * (a + b)
    return x_star, func(x_star)


def numerical_derivative(func: Callable[[float], float], x: float, h: float) -> float:
    """Return a centered finite-difference derivative."""
    return (func(x + h) - func(x - h)) / (2.0 * h)


def first_derivative_from_grid(x_values: list[float], y_values: list[float]) -> list[float]:
    """Return the first derivative on a one-dimensional grid."""
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return []
    if len(x_values) == 2:
        dx = x_values[1] - x_values[0]
        slope = (y_values[1] - y_values[0]) / dx
        return [slope, slope]

    out = [0.0] * len(x_values)
    h_left = x_values[1] - x_values[0]
    h_right = x_values[-1] - x_values[-2]
    out[0] = (-3.0 * y_values[0] + 4.0 * y_values[1] - y_values[2]) / (2.0 * h_left)
    out[-1] = (3.0 * y_values[-1] - 4.0 * y_values[-2] + y_values[-3]) / (2.0 * h_right)

    for index in range(1, len(x_values) - 1):
        dx = x_values[index + 1] - x_values[index - 1]
        out[index] = (y_values[index + 1] - y_values[index - 1]) / dx
    return out


def second_derivative_from_grid(x_values: list[float], y_values: list[float]) -> list[float]:
    """Return the second derivative on a one-dimensional grid."""
    if len(x_values) != len(y_values) or len(x_values) < 3:
        return []

    out = [0.0] * len(x_values)
    h_left = x_values[1] - x_values[0]
    h_right = x_values[-1] - x_values[-2]
    out[0] = (y_values[0] - 2.0 * y_values[1] + y_values[2]) / (h_left * h_left)
    out[-1] = (y_values[-1] - 2.0 * y_values[-2] + y_values[-3]) / (h_right * h_right)

    for index in range(1, len(x_values) - 1):
        h = 0.5 * (x_values[index + 1] - x_values[index - 1])
        out[index] = (y_values[index + 1] - 2.0 * y_values[index] + y_values[index - 1]) / (h * h)
    return out


def solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a small dense linear system with Gaussian elimination."""
    size = len(rhs)
    augmented = [row[:] + [value] for row, value in zip(matrix, rhs)]

    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1.0e-20:
            return [0.0] * size
        if pivot != col:
            augmented[col], augmented[pivot] = augmented[pivot], augmented[col]

        pivot_value = augmented[col][col]
        for j in range(col, size + 1):
            augmented[col][j] /= pivot_value

        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            for j in range(col, size + 1):
                augmented[row][j] -= factor * augmented[col][j]

    return [augmented[index][size] for index in range(size)]


def smooth_local_polynomial(
    x_values: list[float],
    y_values: list[float],
    window_size: int,
    degree: int,
) -> list[float]:
    """Smooth a one-dimensional curve with local polynomial fits."""
    values, _, _ = local_polynomial_regression(
        x_values,
        y_values,
        window_size,
        degree,
    )
    return values


def local_polynomial_regression(
    x_values: list[float],
    y_values: list[float],
    window_size: int,
    degree: int,
) -> tuple[list[float], list[float], list[float]]:
    """Return local-polynomial value, first derivative, and second derivative."""
    if len(x_values) != len(y_values) or not x_values:
        return [], [], []
    if window_size < 3 or degree < 0 or len(x_values) < degree + 1:
        zeros = [0.0 for _ in x_values]
        return y_values[:], zeros[:], zeros[:]

    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    smoothed: list[float] = []
    first_derivative: list[float] = []
    second_derivative: list[float] = []

    for index in range(len(x_values)):
        left = max(0, index - half)
        right = min(len(x_values), index + half + 1)
        if right - left < degree + 1:
            if left == 0:
                right = min(len(x_values), degree + 1)
            else:
                left = max(0, len(x_values) - degree - 1)

        x_local = [x_values[item] - x_values[index] for item in range(left, right)]
        y_local = [y_values[item] for item in range(left, right)]

        dim = degree + 1
        normal = [[0.0 for _ in range(dim)] for _ in range(dim)]
        rhs = [0.0 for _ in range(dim)]

        for x_value, y_value in zip(x_local, y_local):
            powers = [1.0]
            for _ in range(1, dim):
                powers.append(powers[-1] * x_value)
            for row in range(dim):
                rhs[row] += powers[row] * y_value
                for col in range(dim):
                    normal[row][col] += powers[row] * powers[col]

        coeffs = solve_linear_system(normal, rhs)
        smoothed.append(coeffs[0])
        first_derivative.append(coeffs[1] if len(coeffs) >= 2 else 0.0)
        second_derivative.append(2.0 * coeffs[2] if len(coeffs) >= 3 else 0.0)

    return smoothed, first_derivative, second_derivative
