#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

# Функція для обчислення значення функції f(x)
def f(x):
    return 2 * x**7 + 3 * x**4 + 2 * x**2 + 2

# Функція для обчислення значення полінома на основі отриманих коефіцієнтів
def polynomial(x, coeffs):
    result = 0
    for i, coeff in enumerate(coeffs):
        result += coeff * x**(len(coeffs)-i-1)
    return result

# Генерація 5 рівномірно розподілених точок на проміжку [0, 4]
x_values = np.linspace(0, 4, 5)

# Обчислюємо значення функції в цих точках
function_values = [f(x) for x in x_values]

# Створимо систему рівнянь для полінома 4-го ступеня (для 5 точок)
# Поліном виглядає так: P(x) = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4

# Створюємо матрицю для полінома та вектор значень функції
A = np.array([x**np.arange(5) for x in x_values])
b = np.array(function_values)

# Розв'язуємо систему для коефіцієнтів полінома
coefficients = np.linalg.solve(A, b)

# Обчислюємо значення полінома для всіх точок на графіку
x_plot = np.linspace(0, 4, 500)  # Для побудови графіка
y_plot_func = f(x_plot)
y_plot_poly = [polynomial(x, coefficients) for x in x_plot]

# Обчислюємо відхилення між функцією та поліномом на точках апроксимації
polynomial_values = [polynomial(x, coefficients) for x in x_values]
deviations = [abs(fv - pv) for fv, pv in zip(function_values, polynomial_values)]

# Найбільше відхилення
max_deviation = max(deviations)

# Виведення результату
print(f"Найбільше відхилення: {max_deviation}")

# Побудова графіка
plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_plot_func, label='Оригінальна функція: $2x^7 + 3x^4 + 2x^2 + 2$', color='blue')
plt.plot(x_plot, y_plot_poly, label='Наближена функція (поліном)', linestyle='--', color='red')
plt.scatter(x_values, function_values, color='green', zorder=5, label="Точки апроксимації")
plt.legend()
plt.title("Функція та її наближення поліномом 4-го ступеня")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()



# In[ ]:




