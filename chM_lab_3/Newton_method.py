#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Визначення функцій і їх похідних для системи
def f1(x, y):
    return np.tan(x * y + 0.1) - x**2

def f2(x, y):
    return x**2 + 2 * y**2 - 1

# Похідні для Якобіана
def df1_dx(x, y):
    return y / (np.cos(x * y + 0.1))**2 - 2 * x

def df1_dy(x, y):
    return x / (np.cos(x * y + 0.1))**2

def df2_dx(x, y):
    return 2 * x

def df2_dy(x, y):
    return 4 * y

# Якобіан системи
def jacobian(x, y):
    return np.array([[df1_dx(x, y), df1_dy(x, y)],
                     [df2_dx(x, y), df2_dy(x, y)]])

# Метод Ньютона для 5 ітерацій
def newton_method(x0, y0, num_iterations=5, epsilon=1e-6):
    x, y = x0, y0
    results = []
    
    for k in range(num_iterations):
        # Вектор функцій
        F = np.array([f1(x, y), f2(x, y)])
        
        # Якобіан
        J = jacobian(x, y)
        
        # Виведення матриці Якобі, Xk і F(Xk)
        print(f"\nІтерація {k+1}:")
        print(f"Якобіан J(x, y):\n{J}\n")
        print(f"Xk:\n[{x}, {y}]\n")
        print(f"F(Xk):\n{F}\n")
        
        # Вирішення системи лінійних рівнянь для корекції
        delta = np.linalg.solve(J, F)
        
        # Оновлення значень x та y
        x = x - delta[0]
        y = y - delta[1]
        
        # Запис результатів
        results.append([x, y])
        
        # Перевірка на збіжність
        if np.linalg.norm(delta) < epsilon:
            break
    
    return results

# Початкові значення для x і y
x0, y0 = 1, 1

# Виконання методу Ньютона
iterations = newton_method(x0, y0, num_iterations=5)

# Створення таблиці результатів
df_iterations = pd.DataFrame(iterations, columns=["x", "y"])

# Виведення результатів у вигляді таблиці
print("\nРезультати ітерацій:")
print(df_iterations)


# In[ ]:




