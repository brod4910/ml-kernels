import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from typing import Tuple


@dataclass
class dim3:
    x: int = 1
    y: int = 1
    z: int = 1


def ceil_div(m, n):
    return int((m + n - 1) / n)


class GEMM:
    def __init__(self, M = 8, N = 8, K = 8, BLOCK_SIZE = 4):
        self.M = M
        self.N = N
        self.K = K
        self.BLOCK_SIZE = BLOCK_SIZE
        self.A = np.zeros((M, K), dtype=np.uint32)
        self.B = np.zeros((K, N), dtype=np.uint32)
        self.C = np.zeros((M, N), dtype=np.uint32)

        fig, axs = plt.subplots(2, 3, figsize=(10, 10))
        axs[0, 0].set_title("Matrix A")
        axs[0, 2].set_title("Matrix B")
        axs[1, 1].set_title("Matrix C")

        self.axs = axs
        self.fig = fig

        self.cbar = axs[0, 1].imshow(np.full_like(self.A, K), cmap='PuOr', vmin=0, vmax=K)
        plt.colorbar(self.cbar, ax=axs[0, 0])
        plt.colorbar(self.cbar, ax=axs[0, 2])
        plt.colorbar(self.cbar, ax=axs[1, 1])

        fig.delaxes(axs[0, 1])
        fig.delaxes(axs[1, 0])
        fig.delaxes(axs[1, 2])
        self.show()

        self.grid_dim = dim3(ceil_div(M, BLOCK_SIZE), ceil_div(N, BLOCK_SIZE))
        self.block_dim = dim3(BLOCK_SIZE**2)

    def start(self):
        pass

    def update(self, dims: Tuple[dim3, dim3, dim3, int]):
        block_dim, block_idx, thread_idx, k = dims

        x = block_idx.x * self.BLOCK_SIZE + (int(thread_idx.x / self.BLOCK_SIZE))
        y = block_idx.y * self.BLOCK_SIZE + (thread_idx.x % self.BLOCK_SIZE)

        self.A[x, k] += 1
        self.B[k, y] += 1
        self.C[x, y] += 1

        self.show()

    def show(self):
        self.axs[0, 0].imshow(self.A, cmap='PuOr')
        self.axs[0, 2].imshow(self.B, cmap='PuOr')
        self.axs[1, 1].imshow(self.C, cmap='PuOr')

    def generate_frames(self):
        for gx in range(self.grid_dim.x):
            for gy in range(self.grid_dim.y):
                for bx in range(self.block_dim.x):
                    for by in range(self.block_dim.y):
                        for k in range(self.K):
                            yield (self.block_dim, dim3(gx, gy), dim3(bx, by), k)
                        

if __name__ == "__main__":
    gemm = GEMM(4, 4, 4, 2)

    anim = FuncAnimation(gemm.fig, 
                         gemm.update, 
                         frames=gemm.generate_frames, 
                         cache_frame_data=False, 
                         repeat=False,
                         init_func=gemm.start)
    anim.save('coalescing.gif', writer='Pillow', fps=1)
