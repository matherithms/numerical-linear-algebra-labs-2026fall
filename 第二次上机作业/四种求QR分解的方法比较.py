import numpy as np


def CGS(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        # 所有投影系数均使用原始列 A[:, j] 计算
        for i in range(j):
            R[i, j] = Q[:, i] @ A[:, j]

        for i in range(j):
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def MGS(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        # 每次使用当前残差 v 计算投影，并立即更新 v
        for i in range(j):
            R[i, j] = Q[:, i] @ v
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def MGS2(A):
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in range(n):
        v = A[:, j].copy()

        # 连续进行两遍 MGS，第二遍的投影系数需要累加
        for p in range(2):
            for i in range(j):
                c = Q[:, i] @ v
                v = v - c * Q[:, i]
                R[i, j] = R[i, j] + c

        # 两遍正交化完成后再归一化
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R


def Householder(A):
    m, n = A.shape
    A_k = A.copy()
    vectors = []

    # Python 下标从 0 开始
    for k in range(n):
        x = A_k[k:, k].copy()
        e1 = np.zeros(m - k)
        e1[0] = 1

        # 选择符号，避免构造 v 时发生相减消去
        sign = 1 if x[0] >= 0 else -1
        y = -sign * np.linalg.norm(x) * e1
        v = x - y
        vectors.append(v)

        # H_k = I - 2 * v_k @ v_k.T / (v_k.T @ v_k)
        # v_k 为 v 前面补 k 个零得到的完整反射向量
        # 这里只更新受影响的子矩阵，不显式构造 H_k
        A_k[k:, k:] = A_k[k:, k:] - 2 * np.outer(
            v, v @ A_k[k:, k:]
        ) / (v @ v)

    R = np.triu(A_k[:n, :])

    # 构造薄 Q：Q = H_1 @ H_2 @ ... @ H_n @ [I; 0]
    # 从 [I; 0] 开始，按逆序逐次左乘反射矩阵
    Q = np.eye(m, n)
    for k in range(n - 1, -1, -1):
        v = vectors[k]
        Q[k:, :] = Q[k:, :] - 2 * np.outer(
            v, v @ Q[k:, :]
        ) / (v @ v)

    return Q, R


def generate_matrix(m, n, condition_number, rng):
    """
    构造 A = U @ Sigma @ V.T。
    U 的列正交，V 为正交矩阵。
    奇异值从 1 到 1 / condition_number 按等比数列排列，
    从而在精确算术下使 A 的二范数条件数为 condition_number。
    """
    U, _ = np.linalg.qr(
        rng.standard_normal((m, n)), mode='reduced'
    )
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))

    singular_values = np.geomspace(1, 1 / condition_number, n)
    Sigma = np.diag(singular_values)

    return U @ Sigma @ V.T


def orthogonality_error(Q):
    n = Q.shape[1]
    return np.linalg.norm(Q.T @ Q - np.eye(n), ord='fro')


m = 500
n = 20
rng = np.random.default_rng(2026)

# 在 10^1 到 10^16 之间选取 10 个不同的目标条件数
condition_numbers = np.logspace(1, 16, 10)
methods = [CGS, MGS, MGS2, Householder]

print("Orthogonality error: ||Q.T @ Q - I||_F")
print()

print(
    f'{"No.":>4} '
    f'{"Target cond":>14} '
    f'{"Actual cond":>14}'
    f'{"CGS":>14}'
    f'{"MGS":>14}'
    f'{"MGS2":>14}'
    f'{"Householder":>14}'
)

for number, condition_number in enumerate(condition_numbers, start=1):
    A = generate_matrix(m, n, condition_number, rng)
    actual_condition_number = np.linalg.cond(A)

    # 对同一个矩阵分别使用四种方法
    errors = []
    for method in methods:
        Q, R = method(A)
        errors.append(orthogonality_error(Q))

    print(
        f'{number:>4d} '
        f'{condition_number:>14.4e} '
        f'{actual_condition_number:>14.4e}'
        + ''.join(f'{error:>14.6e}' for error in errors)
    )