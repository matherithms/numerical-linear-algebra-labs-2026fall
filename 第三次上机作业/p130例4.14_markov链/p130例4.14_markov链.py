import numpy as np


def print_array(name, A):
    """统一格式输出向量或矩阵。"""
    print(name)
    print(np.array2string(
        A,
        formatter={'float_kind': lambda value: f"{value:.4f}"}
    ))
    print()


# 例4.14使用例4.10中的转移概率矩阵
# p_ij 表示从状态 j 转移到状态 i 的概率，每一列的和为 1
P = np.array([
    [1 / 3, 1 / 2, 3 / 4],
    [2 / 3, 0,     0],
    [0,     1 / 2, 1 / 4]
])

n = P.shape[0]
e = np.ones(n)

print_array("Transition probability matrix P:", P)

# P^2 的每个元素均严格大于0, 从而P^3的每个元素也严格大于0
# 因而该 Markov 链不可约、非周期，满足定理4.20的条件
P2 = P @ P
print_array("P^2:", P2)
print("All entries of P^2 are positive:", np.all(P2 > 0))
print()

# 求平稳分布：P @ pi = pi，且 e.T @ pi = 1
# 将 P - I 的最后一行替换为归一化条件
A = P - np.eye(n)
A[-1, :] = e
b = np.zeros(n)
b[-1] = 1
pi = np.linalg.solve(A, b)

print_array("Stationary distribution pi:", pi)


# 理论极限矩阵为 pi @ e.T，每一列均为平稳分布 pi
P_limit = np.outer(pi, e)
print_array("Limit matrix pi @ e.T:", P_limit)

# 从 P^0 = I 开始，逐次左乘 P 得到 P^t
# 用各元素误差绝对值的最大值衡量收敛程度
P_t = np.eye(n)
steps = [1, 2, 5, 10, 20, 50]

print("Convergence error: max_ij |(P^t)_ij - pi_i|")
print(f'{"t":>6} {"Error":>16}')

for t in range(1, max(steps) + 1):
    P_t = P @ P_t

    if t in steps:
        error = np.max(np.abs(P_t - P_limit))
        print(f'{t:>6d} {error:>16.6e}')

print()
print_array(f"P^{max(steps)}:", P_t)
