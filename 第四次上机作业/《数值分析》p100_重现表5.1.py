from decimal import Decimal, localcontext
from itertools import zip_longest


def df(x):
    """计算 f(x) = -x exp(-x) 的一阶导数。"""
    return (x - 1) * (-x).exp()


def format_error(error):
    """按表 5.1 的形式输出误差。"""
    if error is None:
        return ""
    if error in (Decimal("1"), Decimal("0.5")):
        return f"{error:.1f}"
    if error >= Decimal("0.1"):
        return f"{error:.8f}"

    exponent = error.adjusted() + 1
    return f"{error.scaleb(-exponent):.8f}e{exponent:+d}"


with localcontext() as ctx:
    # 保留足够的精度，以便输出接近 1 时的迭代误差。
    ctx.prec = 40
    tol = Decimal("1e-11")
    x1, x2 = Decimal("0"), Decimal("0.5")

    # Newton 法中 f'(x) / f''(x) 可约去 exp(-x)。
    x = x1
    newton_errors = [1 - x]
    while abs(df(x)) > tol:
        x -= (x - 1) / (2 - x)
        newton_errors.append(1 - x)

    # 按表 5.1 的首两行，割线法从 0 和 0.5 开始。
    secant_errors = [1 - x1, 1 - x2]
    g1, g2 = df(x1), df(x2)
    while abs(g2) > tol:
        x = x2 - g2 * (x2 - x1) / (g2 - g1)
        secant_errors.append(1 - x)
        x1, x2 = x2, x
        g1, g2 = g2, df(x2)

    print("表 5.1  Newton 法与割线法的比较")
    print(f'{"":>8} {"Newton 法":>18} {"割线法":>18}')
    for k, (en, es) in enumerate(zip_longest(newton_errors, secant_errors), 1):
        print(f'{"1 - x" + str(k):>8} {format_error(en):>18} {format_error(es):>18}')