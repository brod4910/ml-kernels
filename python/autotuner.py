import math
import argparse


def list_divisors_excluding_self(n):
    return [i for i in range(1, n) if n % i == 0]


def greatest_divisor(n):
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return i


def calculate_warp_tiling_paramters(M, N, K, max_threads_per_block=128, warp_size=32):
    """
    Calculate CUDA-specific parameters for matrix multiplication C = A x B.

    Args:
        M (int): Number of rows in matrix A (and C).
        N (int): Number of columns in matrix B (and C).
        K (int): Number of columns in matrix A (and rows in B).
        max_threads_per_block (int): Maximum threads per block for the CUDA device.
        warp_size (int): Warp size (default is 32 for most CUDA devices).

    Returns:
        dict: A dictionary containing grid and block dimensions, shared memory size, etc.
    """
    # Thread block dimensions
    block_size_x = min(max_threads_per_block, N)
    block_size_y = max(1, max_threads_per_block // block_size_x)

    # Ensure block dimensions align with warp size
    block_size_x = (block_size_x // warp_size) * warp_size
    if block_size_x * block_size_y > max_threads_per_block:
        block_size_y = max_threads_per_block // block_size_x

    # Grid dimensions
    grid_size_x = math.ceil(N / block_size_x)
    grid_size_y = math.ceil(M / block_size_y)

    # Shared memory requirements
    shared_memory_size = (
        (block_size_x + block_size_y) * K * 4
    )  # Assuming float (4 bytes)

    parameters = {
        "block_size": (block_size_x, block_size_y, 1),
        "grid_size": (grid_size_x, grid_size_y, 1),
        "shared_memory_size": shared_memory_size,
    }
    return parameters


"""
Need to find the valid combinations of warp-tiling parameters.
For example:

BM = 64
BN = 32
BK = 8

WM = 32
WN = 16

WGM = 1
WGN = 1

WTM = 16 => 32 / 2 = 16
WTN = 8  => 16 / 8 = 8

Total Elements = 16 * 8 * 2 * 2 = 512

total_elems / WARPSIZE = 16

GD(16) => 8

1D-Thread-Items = 8

TM = 8 / 2 => 4
TN = 8 / 2 => 4

"""


def main():
    parser = argparse.ArgumentParser(
        description="Calculate CUDA-specific parameters for matrix multiplication."
    )
    parser.add_argument(
        "--M", type=int, required=True, help="Number of rows in matrix A (and C)."
    )
    parser.add_argument(
        "--N", type=int, required=True, help="Number of columns in matrix B (and C)."
    )
    parser.add_argument(
        "--K",
        type=int,
        required=True,
        help="Number of columns in matrix A (and rows in B).",
    )
    parser.add_argument(
        "--max_threads",
        type=int,
        default=128,
        help="Maximum threads per block (default: 1024).",
    )
    parser.add_argument(
        "--warp_size", type=int, default=32, help="Warp size (default: 32)."
    )

    args = parser.parse_args()

    cuda_params = calculate_warp_tiling_paramters(
        args.M, args.N, args.K, args.max_threads, args.warp_size
    )

    print("CUDA Parameters for Matrix Multiplication:")
    print(f"Block size (threads per block): {cuda_params['block_size']}")
    print(f"Grid size (blocks per grid): {cuda_params['grid_size']}")
    print(f"Shared memory size (bytes): {cuda_params['shared_memory_size']}")


if __name__ == "__main__":
    main()
