#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Функція для рівняння x^3 - 7x - 6 = 0
def f(x):
    return x**3 - 7*x - 6

# Похідна функції
def df(x):
    return 3*x**2 - 7

# Оцінка апріорної кількості ітерацій
def aprior_iterations(x0, epsilon=1e-3):
    fx0 = f(x0)
    dfx0 = df(x0)
    
    if abs(dfx0) < 1e-5:
        print("Попередження: похідна надто мала для точного обчислення.")
        return None
    
    iterations = int(np.ceil(np.log(epsilon) / np.log(abs(fx0) / abs(dfx0))))
    return iterations

# Модифікований метод Ньютона з таблицею результатів
def newton_method_with_table(f, df, x0, epsilon=1e-3, max_iter=100):
    iterations_needed = aprior_iterations(x0, epsilon)
    if iterations_needed is not None:
        print(f"Апріорна кількість ітерацій: {iterations_needed}")
    
    results = []
    x = x0
    aposterior_iterations = 0  # Лічильник апостеріорних ітерацій
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        x_new = x - fx / dfx  # Формула методу Ньютона
        results.append([i+1, x_new, fx])  # Додаємо ітерацію, поточне значення та значення функції
        
        aposterior_iterations += 1  # Збільшуємо лічильник апостеріорних ітерацій
        
        if abs(fx) < epsilon:
            df_results = pd.DataFrame(results, columns=["Ітерація", "Наближення", "f(x)"])
            print(df_results)
            print(f"Розв'язок рівняння: x = {x_new}, f(x) = {fx}")
            print(f"Апостеріорна кількість ітерацій: {aposterior_iterations}")
            return x_new
        
        x = x_new
    
    print("Не вдалося знайти розв'язок.")
    print(f"Апостеріорна кількість ітерацій: {aposterior_iterations}")
    return None

# Основна програма
x0 = 2.5  # Початкове значення
epsilon = float(input("Введіть точність (наприклад, 1e-3): "))  # Точність (похибка)

# Розрахунок кількості ітерацій
iterations_needed = aprior_iterations(x0, epsilon)
print(f"Апріорна кількість ітерацій: {iterations_needed}")

# Виконання методу Ньютона
print("Метод Ньютона:")
newton_method_with_table(f, df, x0, epsilon)


# In[ ]:




