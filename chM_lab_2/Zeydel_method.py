#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

def seidel_method(A, b, epsilon=1e-3, max_iterations=1000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iterations):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            return x_new, k + 1
        
        x = x_new
    
    return x, max_iterations  # Return after max_iterations if convergence is not met

def main_seidel():
    # Get user input for precision
    epsilon = float(input("Enter the precision ε (e.g., 1e-3): "))
    
    # Coefficients matrix (A) and right-hand side vector (b)
    A = np.array([[5, 1, 1, 0],
                  [1, 2, 0, 0],
                  [1, 0, 4, 2],
                  [0, 2, 0, 3]])
    b = np.array([10, 5, 21, 18])
    
    # Solve the system using the Seidel method
    solution, iterations = seidel_method(A, b, epsilon)
    
    # Prepare the results for displaying in a table
    results = {
        "X1": [solution[0]],
        "X2": [solution[1]],
        "X3": [solution[2]],
        "X4": [solution[3]],
        "Iterations": [iterations]
    }

    # Creating a pandas DataFrame to display results as a table
    df_results = pd.DataFrame(results)
    return df_results

# Run the main function for Seidel method solution and display the result
df_results = main_seidel()
df_results


# In[ ]:




