#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

def tridiagonal_algorithm(a, b, c, d, epsilon=1e-3):
    n = len(d)
    x = np.zeros(n)

    # Modify the right-hand side (d) using the coefficients a, b, and c
    for i in range(1, n):
        factor = a[i] / b[i-1]
        b[i] -= factor * c[i-1]
        d[i] -= factor * d[i-1]

    # Back substitution
    x[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]

    return x

def main_tridiagonal():
    # Get user input for precision
    epsilon = float(input("Enter the precision ε (e.g., 1e-3): "))
    
    # Coefficients for the system of equations
    a = np.array([0, 2, 0])  # Subdiagonal (a[i] corresponds to A[i+1,i])
    b = np.array([1, 2, 3])  # Diagonal (b[i] corresponds to A[i,i])
    c = np.array([2, 4, 0])  # Superdiagonal (c[i] corresponds to A[i,i+1])
    d = np.array([5, 22, 18])  # Right-hand side vector
    
    # Solve the system using the tridiagonal algorithm
    solution = tridiagonal_algorithm(a, b, c, d, epsilon)

    # Prepare the results for displaying in a table
    results = {
        "X1": [solution[0]],
        "X2": [solution[1]],
        "X3": [solution[2]]
    }

    # Creating a pandas DataFrame to display results as a table
    df_results = pd.DataFrame(results)
    return df_results

# Run the main function for tridiagonal system solution and display the result
df_results = main_tridiagonal()
df_results


# In[ ]:




