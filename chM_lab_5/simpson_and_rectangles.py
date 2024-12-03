#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Функція для обчислення значення функції f(x)
def f(x):
    return 2 * x**7 + 3 * x**4 + 2 * x**2 + 2

# Формула середніх прямокутників
def middle_rectangles(a, b, n):
    dx = (b - a) / n
    integral = 0
    for i in range(n):
        x_mid = a + (i + 0.5) * dx  # середина кожного підінтервалу
        integral += f(x_mid)
    return integral * dx

# Формула Сімпсона
def simpsons_rule(a, b, n):
    if n % 2 == 1:  # Переконуємося, що n парне
        n += 1
    dx = (b - a) / n
    integral = f(a) + f(b)
    for i in range(1, n, 2):  # Непарні точки
        integral += 4 * f(a + i * dx)
    for i in range(2, n-1, 2):  # Парні точки
        integral += 2 * f(a + i * dx)
    return integral * dx / 3

# Виконання обчислень для обох методів
a = 0
b = 4
n = 10  # Кількість підінтервалів

# Обчислення за методом середніх прямокутників
rect_integral = middle_rectangles(a, b, n)
print(f"Інтеграл за методом середніх прямокутників: {rect_integral}")

# Обчислення за методом Сімпсона
simp_integral = simpsons_rule(a, b, n)
print(f"Інтеграл за методом Сімпсона: {simp_integral}")


# In[ ]:




