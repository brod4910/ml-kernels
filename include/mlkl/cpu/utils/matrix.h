#include <cstddef>

namespace mlkl::cpu::utils {
/**
 * @brief Asserts the correctness of a matrix by comparing it to a reference matrix.
 *
 * This function checks if all elements of a matrix are within a specified tolerance 
 * of the corresponding elements in a reference matrix. The comparison uses the 
 * formula: |matrix[i] - ref_matrix[i]| <= epsilon.
 *
 * @param matrix Pointer to the 1D array representing the matrix to check.
 * @param ref_matrix Pointer to the 1D array representing the reference matrix.
 * @param M Number of rows in the matrices.
 * @param N Number of columns in the matrices.
 * @param epsilon Tolerance for element-wise comparison (default: 1e-6).
 * @return true if all elements are within the tolerance, false otherwise.
 */
bool assert_correctness(float *matrix, float *ref_matrix, size_t M, size_t N, float epsilon = 1e-6);

/**
 * @brief Prints the contents of a matrix to the console.
 *
 * This function displays the elements of a matrix in a human-readable format, 
 * where rows and columns are represented in a grid-like structure. The matrix 
 * is assumed to be stored in a row-major 1D array format.
 *
 * @param matrix Pointer to the 1D array representing the matrix.
 * @param M Number of rows in the matrix.
 * @param N Number of columns in the matrix.
 */
void print_matrix(const float *matrix, size_t M, size_t N);
}// namespace mlkl::cpu::utils