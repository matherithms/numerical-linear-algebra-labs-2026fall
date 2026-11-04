import numpy as np


def print_array(name, A):
    """统一格式输出向量或矩阵。"""
    print(name)
    print(np.array2string(
        A,
        formatter={'float_kind': lambda value: f"{value:.4f}"}
    ))
    print()


# 图4.7的邻接矩阵
# a_ij = 1 表示节点 v_j 指向节点 v_i
A = np.array([
    [0, 0, 1, 0, 1, 1],
    [1, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0]
])

n = A.shape[0]
alpha = 0.85

# r_i 为指向节点 v_i 的节点数量
# c_j 为节点 v_j 所指向的节点数量
r = np.sum(A, axis=1)
c = np.sum(A, axis=0)

print_array("Adjacency matrix A:", A)
print_array("Row sums r:", r)
print_array("Column sums c:", c)

# 按教材公式逐列构造转移概率矩阵
P = np.zeros((n, n))

for j in range(n):
    if c[j] != 0:
        P[:, j] = alpha * A[:, j] / c[j] + (1 - alpha) / n
    else:
        # 无出链节点：等概率转移到所有节点
        P[:, j] = 1 / n

print_array("Transition probability matrix P:", P)
print()

# P 的每个元素均严格大于0，因此对应的 Markov 链不可约、非周期
# 从均匀初始分布出发，用幂法计算唯一的平稳分布
pi = np.ones(n) / n
tol = 1e-12
max_iter = 10000

for t in range(1, max_iter + 1):
    pi_new = P @ pi

    # 相邻两次迭代之差的无穷范数，作为停止条件
    error = np.max(np.abs(pi_new - pi))
    pi = pi_new

    if error < tol:
        break

print_array("Stationary distribution pi:", pi)

# 按平稳概率从大到小排列，Python下标加1得到节点编号
order = np.argsort(-pi)

print("Page ranking:")
print(f'{"Rank":>6} {"Node":>6} {"Probability":>14}')

for rank, j in enumerate(order, start=1):
    print(f'{rank:>6d} {j + 1:>6d} {pi[j]:>14.4f}')