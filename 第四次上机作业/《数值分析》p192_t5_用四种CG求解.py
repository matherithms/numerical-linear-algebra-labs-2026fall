import numpy as np


def f(x):
    u, v = x
    return 100 * (v - u**2)**2 + (1 - u)**2


def grad(x):
    u, v = x
    t = v - u**2
    return np.array([2 * (u - 1) - 400 * u * t, 200 * t])


def line_search(x, d):
    """沿下降方向 d 搜索，返回使 f(x + alpha * d) 最小的正步长。"""
    u, v = x
    p, q = d

    # phi(alpha) = 100(a + b*alpha + c*alpha**2)**2
    #              + (u - 1 + p*alpha)**2
    a = v - u**2
    b = q - 2 * u * p
    c = -p**2

    # 求 phi'(alpha) = 0，系数按降幂排列。
    roots = np.roots([
        400 * c**2,
        600 * b * c,
        200 * (b**2 + 2 * a * c) + 2 * p**2,
        200 * a * b + 2 * (u - 1) * p,
    ])

    # 保留正实根，允许求根时产生微小的虚部。
    candidates = [
        root.real for root in roots
        if root.real > 0 and abs(root.imag) <= 1e-10 * root.real
    ]

    return min(candidates, key=lambda alpha: f(x + alpha * d))


def conjugate_gradient(x, method, tol=1e-16, max_iter=10000):
    """用指定的 beta 公式迭代，返回终点和更新次数。"""
    g = grad(x)
    d = -g

    for k in range(max_iter):
        if f(x) <= tol:
            return x, k

        alpha = line_search(x, d)
        
        x = x + alpha * d
        g_next = grad(x)
        y = g_next - g

        if method == "FR":
            numerator, denominator = g_next @ g_next, g @ g
        elif method == "PRP":
            numerator, denominator = g_next @ y, g @ g
        elif method == "HS":
            numerator, denominator = g_next @ y, d @ y
        elif method == "DY":
            numerator, denominator = g_next @ g_next, d @ y

        beta = numerator / denominator if denominator > 0 else 0.0
        d = -g_next + beta * d
        g = g_next

    return x, max_iter


x1 = np.array([-1.2, 1.0])
print(f'{"Method":<6} {"Steps":>6} {"x[0]":>15} {"x[1]":>15} '
f'{"f(x)":>13} ')

for method in ("FR", "PRP", "HS", "DY"):
    x, steps = conjugate_gradient(x1, method)
    print(f'{method:<6} {steps:>6d} {x[0]:>15.10f} {x[1]:>15.10f} '
    f'{f(x):>13.4e}')
