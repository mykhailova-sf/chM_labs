#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import math

# Степеневий метод для знаходження найменшого власного значення
def power_iteration(A, num_iter=100, epsilon=1e-6):
    n = A.shape[0]
    b_k = np.ones(n)
    for _ in range(num_iter):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        if np.linalg.norm(np.dot(A, b_k) - b_k * b_k1_norm) < epsilon:
            break
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k))
    return eigenvalue, b_k

# Метод обертання Якобі для знаходження всіх власних значень
def jacobi_rotation(A, num_iter=3):
    n = A.shape[0]
    A_k = np.copy(A)
    U = np.eye(n)
    
    # Зберігаємо початкову матрицю A0
    A_0 = np.copy(A)
    
    for k in range(num_iter):
        # Знаходимо найбільший за модулем не діагональний елемент
        max_off_diag = np.unravel_index(np.argmax(np.abs(np.triu(A_k, 1))), A_k.shape)
        i, j = max_off_diag
        
        if A_k[i, i] == A_k[j, j]:
            theta = np.pi / 4
        else:
            theta = 0.5 * np.arctan(2 * A_k[i, j] / (A_k[i, i] - A_k[j, j]))
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        R = np.eye(n)
        R[i, i] = cos_theta
        R[j, j] = cos_theta
        R[i, j] = -sin_theta
        R[j, i] = sin_theta
        
        # Оновлюємо A та U
        A_k = np.dot(np.dot(R.T, A_k), R)
        U = np.dot(U, R)
        
        # Виводимо A_k та U на кожній ітерації
        print(f"Iteration {k + 1}:")
        print(f"Matrix U (Rotation Matrix) at iteration {k + 1}:\n", U)
        print(f"Matrix A_{k + 1} (Updated Matrix) at iteration {k + 1}:\n", A_k)
        print()
    
    eigenvalues = np.diagonal(A_k)
    return eigenvalues, A_k, U, A_0

# Основна функція для розв'язання задачі
def main():
    # Коэффициенты для матриці рівнянь
    A = np.array([[4, 0, 2, 0],
                  [0, 3, 0, 1],
                  [2, 0, 3, 0],
                  [0, 1, 0, 2]])
    
    # Початкова матриця A0
    print("Initial Matrix A0:\n", A)
    
    # Виконання степеневого методу для найменшого власного значення
    eigenvalue_power, eigenvector_power = power_iteration(A)
    print("Eigenvalue (Power Iteration):", eigenvalue_power)
    print("Eigenvector (Power Iteration):", eigenvector_power)
    
    # Виконання методу обертання Якобі для всіх власних значень
    eigenvalues_jacobi, A_jacobi, U_jacobi, A_0 = jacobi_rotation(A, num_iter=3)
    print("Eigenvalues (Jacobi Method):", eigenvalues_jacobi)

# Виконання основної функції
main()


# In[ ]:




