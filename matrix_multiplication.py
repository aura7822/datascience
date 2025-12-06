import numpy as np

A=np.array(
    [1,2,3],
    [4,5,6],
    [7,8,9]
)
B=np.array(
    [9,8,7],
    [6,5,4],
    [3,2,1]
)
#matrix multiplication:
matrix_product = np.dot(A,B)
print("Matrix multiplication result:")
print(matrix_product)

#determinant calculation:
determinant_A = np.linalg.det(A)
determinant_B = np.linalg.det(B)
print("Determinant of matrix A:{determinant_A:.2f}")
print("Determinant of matrix B:{determinant_B:.2f}")



