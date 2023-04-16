# Thanapoom Phatthanaphan
# CWID: 20011296
# CS 556-A
# Part 1: Matrix operations

import numpy as np


def add_matrixes(matrix_a, matrix_b):
    """
    Function to add two 3x3 matrices
    :param matrix_a: a matrix to be added
    :param matrix_b: another matrix to be added
    :return: The addition matrix of matrix_a and matrix_b
    """
    added_matrix = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b)):
            added_matrix[i][j] = matrix_a[i][j] + matrix_b[i][j]

    return added_matrix


def add_matrices_numpy(matrix_a, matrix_b):
    """
    Function to add two 3x3 matrices by using numpy library
    :param matrix_a: a matrix to be added
    :param matrix_b: another matrix to be added
    :return: The addition matrix of matrix_a and matrix_b
    """
    return np.add(matrix_a, matrix_b)


def multiply_matrixes(matrix_a, matrix_b):
    """
    Function to multiply two 3x3 matrices
    :param matrix_a: a matrix to be multiplied
    :param matrix_b: another matrix to be multiplied
    :return: The multiplication matrix of matrix_a and matrix_b
    """
    multiplied_matrix = [[0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]]

    for i in range(len(matrix_a)):
        for j in range(len(matrix_b[0])):
            for k in range(len(matrix_b)):
                multiplied_matrix[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return multiplied_matrix


def multiply_matrices_numpy(matrix_a, matrix_b):
    """
    Function to multiply two 3x3 matrices by using numpy library
    :param matrix_a: a matrix to be multiplied
    :param matrix_b: another matrix to be multiplied
    :return: The multiplication matrix of matrix_a and matrix_b
    """
    return np.dot(matrix_a, matrix_b)


def transpose_matrix(matrix_a):
    """
    Function to transpose a matrix
    :param matrix_a: a matrix to be transposed
    :return: a transposed matrix
    """
    transpose_matrix = []
    for i in range(len(matrix_a[0])):
        row = []
        for j in range(len(matrix_a)):
            row.append(matrix_a[j][i])
        transpose_matrix.append(row)
    return transpose_matrix


def transpose_matrix_numpy(matrix_a):
    """
    Function to transpose a matrix by using numpy library
    :param matrix_a: a matrix to be transposed
    :return: a transposed matrix
    """
    return np.transpose(matrix_a)


def inverse_matrix(matrix_a):
    """
    Function to inverse a matrix by using numpy library
    :param matrix_a: a matrix to be inversed
    :return: an inversed matrix
    """
    return np.linalg.inv(matrix_a)


def inverse_matrix_3x3(matrix_a):
    """
    Function to inverse a 3x3 matrix without using numpy library
    :param matrix_a: a 3x3 matrix to be inversed
    :return: a 3x3 inversed matrix
    """
    # Step 0: Invertibility Test - if the determinant of the matrix is equal to 0, then the matrix is not invertible
    det_matrix = 0
    for i in range(len(matrix_a)):
        for j in range(len(matrix_a[0])):
            det_minor = 0
            if j == 0:
                det_minor += (matrix_a[i + 1][j + 1] * matrix_a[i + 2][j + 2]) - \
                             (matrix_a[i + 2][j + 1] * matrix_a[i + 1][j + 2])
            if j == 1:
                det_minor += (matrix_a[i + 1][j - 1] * matrix_a[i + 2][j + 1]) - \
                             (matrix_a[i + 2][j - 1] * matrix_a[i + 1][j + 1])
            if j == 2:
                det_minor += (matrix_a[i + 1][j - 2] * matrix_a[i + 2][j - 1]) - \
                             (matrix_a[i + 2][j - 2] * matrix_a[i + 1][j - 1])
            det_matrix += ((-1) ** ((i + 1) + (j + 1))) * matrix_a[i][j] * det_minor
        break
    if det_matrix == 0:
        return None

    # Step 1: Minor matrix - a matrix of determinants
    minor_matrix = []
    for i in range(len(matrix_a)):
        row = []
        for j in range(len(matrix_a[0])):
            det_minor = 0
            if i == 0 and j == 0:
                det_minor += (matrix_a[i + 1][j + 1] * matrix_a[i + 2][j + 2]) - \
                             (matrix_a[i + 2][j + 1] * matrix_a[i + 1][j + 2])
            if i == 0 and j == 1:
                det_minor += (matrix_a[i + 1][j - 1] * matrix_a[i + 2][j + 1]) - \
                             (matrix_a[i + 2][j - 1] * matrix_a[i + 1][j + 1])
            if i == 0 and j == 2:
                det_minor += (matrix_a[i + 1][j - 2] * matrix_a[i + 2][j - 1]) - \
                             (matrix_a[i + 2][j - 2] * matrix_a[i + 1][j - 1])
            if i == 1 and j == 0:
                det_minor += (matrix_a[i - 1][j + 1] * matrix_a[i + 1][j + 2]) - \
                             (matrix_a[i + 1][j + 1] * matrix_a[i - 1][j + 2])
            if i == 1 and j == 1:
                det_minor += (matrix_a[i - 1][j - 1] * matrix_a[i + 1][j + 1]) - \
                             (matrix_a[i + 1][j - 1] * matrix_a[i - 1][j + 1])
            if i == 1 and j == 2:
                det_minor += (matrix_a[i - 1][j - 2] * matrix_a[i + 1][j - 1]) - \
                             (matrix_a[i + 1][j - 2] * matrix_a[i - 1][j - 1])
            if i == 2 and j == 0:
                det_minor += (matrix_a[i - 2][j + 1] * matrix_a[i - 1][j + 2]) - \
                             (matrix_a[i - 1][j + 1] * matrix_a[i - 2][j + 2])
            if i == 2 and j == 1:
                det_minor += (matrix_a[i - 2][j - 1] * matrix_a[i - 1][j + 1]) - \
                             (matrix_a[i - 1][j - 1] * matrix_a[i - 2][j + 1])
            if i == 2 and j == 2:
                det_minor += (matrix_a[i - 2][j - 2] * matrix_a[i - 1][j - 1]) - \
                             (matrix_a[i - 1][j - 2] * matrix_a[i - 2][j - 1])
            row.append(det_minor)
        minor_matrix.append(row)

    # Step 2: Cofactors matrix - the minors matrix element-wise multiplied by a grid of alternating +1 nad -1.
    for i in range(len(minor_matrix)):
        for j in range(len(minor_matrix)):
            if (i == 0 and j == 1) or (i == 1 and (j == 0 or j == 2)) or (i == 2 and j == 1):
                minor_matrix[i][j] *= -1

    cofactors_matrix = minor_matrix

    # Step 3: Adjugate matrix - the transpose of the cofactors matrix
    adjugate_matrix = []
    for i in range(len(cofactors_matrix[0])):
        row = []
        for j in range(len(cofactors_matrix)):
            row.append(cofactors_matrix[j][i])
        adjugate_matrix.append(row)

    # Step 4: Inverse matrix - the adjugate matrix divided by the determinant
    inverse_matrix = []
    for i in range(len(adjugate_matrix)):
        row = []
        for j in range(len(adjugate_matrix)):
            row.append(adjugate_matrix[i][j] / det_matrix)
        inverse_matrix.append(row)
    return inverse_matrix

# Test operation
X = [[12,7,3],
    [4,5,6],
    [7,8,9]]

Y = [[5,8,1],
    [6,7,3],
    [4,5,9]]


print("Addition")
print(add_matrixes(X, Y))
print(add_matrices_numpy(X,Y))
print()
print("Multiplication")
print(multiply_matrixes(X, Y))
print(multiply_matrices_numpy(X,Y))
print()
print("Transpose")
print(transpose_matrix(X))
print(transpose_matrix_numpy(X))
print()
print("Inverse")
print(inverse_matrix(X))
print(inverse_matrix_3x3(X))
