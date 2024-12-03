#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

def gaussian_elimination(A, b, epsilon=1e-3):
    n = len(b)
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination
    for i in range(n):
        # Find pivot element
        pivot_row = np.argmax(np.abs(augmented_matrix[i:n, i])) + i
        augmented_matrix[[i, pivot_row]] = augmented_matrix[[pivot_row, i]]

        # Make the diagonal element equal to 1
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]

        # Eliminate below
        for j in range(i + 1, n):
            factor = augmented_matrix[j, i]
            augmented_matrix[j] = augmented_matrix[j] - factor * augmented_matrix[i]

    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.sum(augmented_matrix[i, i+1:n] * x[i+1:n])

    # Compute the determinant
    det_A = np.prod(np.diagonal(augmented_matrix[:, :-1]))

    return x, det_A, augmented_matrix[:, :-1]

def calculate_inverse(A, epsilon=1e-3):
    try:
        return np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return None  # Singular matrix, no inverse

def main():
    # Input system of equations
    A = np.array([[4, 3, 1, 0],
                  [-2, 2, 6, 1],
                  [0, 5, 2, 3],
                  [0, 1, 2, 7]])

    b = np.array([29, 38, 48, 56])

    # Ask user for precision
    epsilon = float(input("Enter the precision ε (e.g., 1e-3): "))

    # Solve system using Gaussian elimination
    x, det_A, augmented_matrix = gaussian_elimination(A, b, epsilon)

    # Calculate inverse matrix
    A_inv = calculate_inverse(A, epsilon)

    # Display results
    print(f"Solution (X1, X2, X3, X4): {x}")
    print(f"Determinant: {det_A}")
    
    if A_inv is not None:
        print("Inverse Matrix:")
        print(A_inv)
    else:
        print("Matrix is singular, no inverse exists.")
    
    return x, det_A, A_inv

if __name__ == "__main__":
    main()


# In[ ]:




