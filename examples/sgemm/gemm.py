import numpy as np


def gemm(M, N, K, m, n, k, alpha, beta):
     a = np.zeros((M, K))
     b = np.zeros((K, N))
     c = np.zeros((M, N))
     a.fill(m)
     b.fill(n)
     c.fill(k)
     return np.einsum(',ij,jk->ik', alpha, a, b) + np.einsum(',ik', beta, c)


print(gemm(32, 32, 32, 1, 2, 0, 1.0, 0.0))
m = np.arange(0, 64).reshape(8, 8)
print(m.T)
