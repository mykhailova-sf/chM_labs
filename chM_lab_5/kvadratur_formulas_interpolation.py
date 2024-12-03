#!/usr/bin/env python
# coding: utf-8

# In[2]:


import copy
import numpy as np
import pandas as pd

def f(x):
    return 2 * x**7 + 3 * x**4 + 2 * x**2 + 2

# Функція для множення поліномів
def mult_pol(a: np.array, b: np.array) -> np.array:
    res = np.zeros(len(a) + len(b) - 1)
    for i in range(len(a)):
        for j in range(len(b)):
            res[i + j] += a[i] * b[j]
    return res

# Функція для інтеграції полінома
def integrate_polynomial(p: np.array) -> np.array:
    for i in range(len(p)):
        p[i] /= i + 1

    p = np.insert(p, 0, 0)
    return p

# Функція для обчислення значення полінома в точці x
def get_value_for_polynomial(p: np.array, x: float):
    res = 0
    for i in range(len(p)):
        res += p[i] * x ** i
    return res

# Функція для обчислення визначеного інтегралу
def get_definitive_integral(integrated_p: np.array, a: int, b: int):
    return get_value_for_polynomial(integrated_p, b) - get_value_for_polynomial(integrated_p, a)

# Функція для отримання полінома Лагранжа
def get_lagrange_polynomial(points, number: int) -> np.array:
    res = [1]
    div = 1
    for i in range(len(points)):
        if i != number:
            res = mult_pol(res, [-points[i], 1])
            div *= points[number] - points[i]

    # Виведення полінома Лагранжа
    print(f"Поліном Лагранжа для точки {number} (l[{number}]):")
    print(f"({print_poly(res)})/{div}")
    for i in range(len(res)):
        res[i] /= div
    return res

# Функція для виведення полінома у вигляді рядка
def print_poly(p):
    terms = []
    for i in range(len(p)):
        if p[i] != 0:
            if i == 0:
                terms.append(f"{p[i]}")
            elif i == 1:
                terms.append(f"{p[i]} * x")
            else:
                terms.append(f"{p[i]} * x^{i}")
    return " + ".join(terms)

# Функція для обчислення наближеного значення
def calculate_approximate_value(w, points):
    approx_f = 0
    for i in range(len(w)):
        approx_f += w[i] * f(points[i])
    return approx_f


# Функція для перевірки алгебраїчної точності (AST)
def calculate_ast(points, w, a, b):
    ans = 0
    while is_accurate(points, w, ans, a, b):
        ans += 1
    return ans

# Функція для тестування полінома
def test_polynomial(points, w, power):
    res = 0
    for i in range(len(w)):
        res += w[i] * pow(points[i], power)
    return res

# Функція для перевірки точності
def is_accurate(points, w, power, a, b):
    test_value = b ** (power + 1) / (power + 1) - a ** (power + 1) / (power + 1)
    poly_value = test_polynomial(points, w, power)
    print(f"Крок {power}:")
    print(f"Інтеграл(x^{power}) на [{a}, {b}] = {test_value}")
    print(f"Сума(w[i] * f[i]): {poly_value}")
    return abs(test_value - poly_value) < 0.000000001

# Основна функція для вирішення задачі
def solve(points):
    polynomials = []
    integrated_polynomials = []
    w = []
    
    print("\nРозрахунок поліномів Лагранжа:")
    for i in range(len(points)):
        l_p = get_lagrange_polynomial(points, i)
        polynomials.append(l_p)
        integrated_polynomials.append(integrate_polynomial(copy.deepcopy(l_p)))
        current_w = get_definitive_integral(integrated_polynomials[-1], points[0], points[-1])
        w.append(current_w)
    
    print("\nТаблиця інтегрованих поліномів:")
    print(pd.DataFrame(integrated_polynomials, columns=[f"x^{i}" for i in range(len(points) + 1)]))
    
    print("\nТаблиця значень w:")
    print(pd.DataFrame([w], columns=[f"w[{i}]" for i in range(len(points))]))

    print("\nПошук алгебраїчного степеня точності (AST):")
    ast = calculate_ast(points, w, points[0], points[-1]) - 1
    print(f"Алгебраїчний степінь точності (AST) = {ast}")
    
    approx_value = calculate_approximate_value(w, points)
    print(f"Наближене значення інтегралу: {approx_value}")
    
    return w, ast, approx_value

# Виклик функції з точками на проміжку [0, 4]
x_points = np.array([0, 1, 2, 3, 4])  # Точки для побудови квадратурної формули
solve(x_points)


# In[ ]:




