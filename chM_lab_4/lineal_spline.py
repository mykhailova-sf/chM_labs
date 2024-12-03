#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# Оригінальна функція
def f(x):
    return 2 * x**7 + 3 * x**4 + 2 * x**2 + 2

# Лінійний сплайн (інтерполяція між точками)
def linear_spline(x, x_points, y_points):
    for i in range(len(x_points)-1):
        if x >= x_points[i] and x <= x_points[i+1]:
            return y_points[i] + (y_points[i+1] - y_points[i]) * (x - x_points[i]) / (x_points[i+1] - x_points[i])

# Визначаємо вузли з кроком 0.5
x_points = np.arange(0, 4.5, 0.5)
y_points = f(x_points)

# Побудова лінійного сплайну для побудови графіка
x_vals = np.linspace(0, 4, 500)
y_vals_spline = np.array([linear_spline(x, x_points, y_points) for x in x_vals])

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(x_vals, f(x_vals), label='Оригінальна функція', color='blue')
plt.plot(x_vals, y_vals_spline, label='Лінійний сплайн', linestyle='--', color='red')
plt.scatter(x_points, y_points, color='green', label="Точки апроксимації")
plt.title('Лінійний сплайн для функції')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Виведення коефіцієнтів сплайну
print(f"Вузли (x): {x_points}")
print(f"Значення функції у вузлах (y): {y_points}")


# In[ ]:




