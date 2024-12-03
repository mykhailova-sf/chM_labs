#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Оригінальна функція
def f(x):
    return 2 * x**7 + 3 * x**4 + 2 * x**2 + 2

# Похідна функції f(x) (для використання умов на краях)
def f_prime(x):
    return 14 * x**6 + 12 * x**3 + 4 * x

# Точки для побудови сплайну
x_points = np.array([0, 2, 4])
y_points = f(x_points)

# Похідні на краях
f_prime_0 = f_prime(0)  # Похідна в точці x=0
f_prime_4 = f_prime(4)  # Похідна в точці x=4

# Створення матриці системи рівнянь для квадратичного сплайну
# Матриця для сплайну з двох сегментів
A = np.array([
    [1, 0, 0, 0, 0],   # для першої точки x=0
    [0, 1, 0, 0, 0],   # для середини x=2 (умови неперервності)
    [0, 0, 1, 0, 0],   # для кінця x=4
    [0, 0, 0, 1, 0],   # умови на похідні в x=0
    [0, 0, 0, 0, 1]    # умови на похідні в x=4
])

# Вектор для правих частин рівнянь
b = np.array([
    y_points[0],          # значення функції в точці x=0
    y_points[1],          # значення функції в точці x=2
    y_points[2],          # значення функції в точці x=4
    f_prime_0,            # значення похідної на краю x=0
    f_prime_4             # значення похідної на краю x=4
])

# Розв'язуємо систему рівнянь для коефіцієнтів сплайну
coefficients = np.linalg.solve(A, b)

# Розв'язок системи дасть коефіцієнти для двох квадратичних поліномів
a0, a1, a2, b0, b1 = coefficients

# Створюємо функцію для побудови сплайну
def spline(x):
    if x < 2:
        return a0 + a1*x + a2*x**2
    else:
        return b0 + b1*x

# Обчислюємо значення сплайну для побудови графіка
x_vals = np.linspace(0, 4, 500)
y_vals_spline = np.array([spline(x) for x in x_vals])

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), label='Оригінальна функція', color='blue')
plt.plot(x_vals, y_vals_spline, label='Квадратичний сплайн', linestyle='--', color='red')
plt.scatter(x_points, y_points, color='green', label="Точки апроксимації")
plt.title('Квадратичний сплайн для функції')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Виведення коефіцієнтів сплайну
print(f"Коефіцієнти сплайну: a0 = {a0}, a1 = {a1}, a2 = {a2}, b0 = {b0}, b1 = {b1}")


# In[ ]:




