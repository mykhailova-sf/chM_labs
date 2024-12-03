#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Функція для рівняння
def f(x):
    return x**3 - 6*x**2 + 5*x + 12

# Функція для перетворення рівняння у форму методу простої ітерації
def phi(x):
    return (6 * x**2 - 5 * x - 12) / (x**2)

# Оцінка апріорної кількості ітерацій
def aprior_iterations(x0, epsilon=1e-3):
    fx0 = f(x0)
    dfx0 = 3 * x0**2 - 12 * x0 + 5  # Похідна функції
    if abs(dfx0) < 1e-5:  # Якщо похідна дуже мала
        print("Попередження: похідна дуже мала для точного обчислення.")
        return None
    try:
        # Формула для апріорної кількості ітерацій
        iterations = int(np.ceil(np.log(epsilon) / np.log(abs(fx0) / abs(dfx0))))
    except ZeroDivisionError:
        print("Помилка: ділення на нуль при розрахунку апріорної кількості ітерацій.")
        return None
    return iterations

# Метод простої ітерації
def iteration_method(phi, x0, epsilon=1e-3, max_iter=100):
    results = []
    x = x0
    for i in range(max_iter):
        x_new = phi(x)
        fx = f(x_new)
        results.append([i + 1, x_new, fx])
        
        # Перевірка умови зупинки
        if abs(fx) < epsilon:
            break
        x = x_new

    df_results = pd.DataFrame(results, columns=["Ітерація", "Наближення", "f(x)"])
    print(df_results)
    return x_new, i + 1  # Повертаємо також кількість ітерацій

# Основна програма
x0 = 2.0
epsilon = float(input("Введіть точність (наприклад, 1e-3): "))
iterations_needed = aprior_iterations(x0, epsilon)

if iterations_needed is not None:
    print(f"Апріорна кількість ітерацій: {iterations_needed}")
    print("Метод простої ітерації:")
    root, iterations_performed = iteration_method(phi, x0, epsilon)
    print(f"Розв'язок рівняння: x = {root}, f(x) = {f(root)}")
    print(f"Апостеріорна кількість ітерацій: {iterations_performed}")


# In[ ]:




