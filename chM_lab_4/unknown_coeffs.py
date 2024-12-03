#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# Функція для обчислення значення f(x)
def f(x):
    return 2 * x**7 + 3 * x**4 + 2 * x**2 + 2

# Функція для отримання коефіцієнтів (x^i) для кожного вузла
def get_coefficients(x: float, number_of_points: int) -> np.array:
    ans = np.zeros(number_of_points+1)
    for i in range(number_of_points+1):
        ans[i] = x ** i
    return ans

# Функція для формування рівняння для кожного вузла
def get_equation(x: float, number_of_points: int):
    return get_coefficients(x, number_of_points), f(x)

# Функція для побудови матриці рівнянь та вектора значень
def get_matrix_equation(l: float, r: float, number_of_points: int) -> np.array:
    a = list()
    ans = list()
    h = (r - l)/number_of_points
    i = l
    while i <= r:
        cur = get_equation(i, number_of_points)
        ans.append(cur[1])
        a.append(cur[0])
        i += h
    return a, ans

# Метод Гаусса для розв'язку системи рівнянь
def gauss_elimination(a: np.array, b: np.array):
    n = len(b)
    # Прямий хід
    for i in range(n):
        # Знаходимо максимум для покращення стійкості
        max_row = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[[i, max_row]] = a[[max_row, i]]  # міняємо рядки місцями
        b[i], b[max_row] = b[max_row], b[i]

        for j in range(i+1, n):
            factor = a[j][i] / a[i][i]
            a[j][i:] -= factor * a[i][i:]
            b[j] -= factor * b[i]

    # Зворотний хід
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(a[i, i+1:], x[i+1:])) / a[i, i]
    return x

# Основна функція для розв'язку
def solve(l: float, r: float, number_of_points: int):
    a, b = get_matrix_equation(l, r, number_of_points)
    a = np.array(a)
    b = np.array(b)

    # Виводимо початкові дані
    print("Матриця невизначених коефіцієнтів A: ")
    print(pd.DataFrame(a))  # Використовуємо pandas для красивого виведення

    print(f"Вектор значень функції: {b}")

    # Розв'язуємо систему методом Гаусса
    coefficients_gauss = gauss_elimination(a, b)
    print("Розв'язок системи (коефіцієнти полінома):", coefficients_gauss)

    # Перевіряємо значення функції на виході
    y_calculated = np.dot(a, coefficients_gauss)
    print("Перевірка значень функції:", y_calculated)

    # Виводимо результуючий поліном
    print("Результуючий поліном: ", end="")
    for i in range(len(coefficients_gauss)):
        if i > 0:
            if coefficients_gauss[i] > 0:
                print(" + ", end="")
        print(f"{coefficients_gauss[i]} * x^{i}", end="")

    # Перевіряємо значення полінома у граничних точках
    x_values = np.array([l, r])
    f_values = f(x_values)
    p_values = np.polyval(coefficients_gauss[::-1], x_values)  # оскільки np.polyval працює від старшого до найменшого степеня

    print(f"\nПеревірка значень функції у граничних точках:")
    print(f"f({l}) = {f_values[0]}, P({l}) = {p_values[0]}")
    print(f"f({r}) = {f_values[1]}, P({r}) = {p_values[1]}")

# Запуск розв'язку для функції 2*x^7 + 3*x^4 + 2*x^2 + 2 на проміжку [0, 4]
solve(0, 4, 4)


# In[ ]:




