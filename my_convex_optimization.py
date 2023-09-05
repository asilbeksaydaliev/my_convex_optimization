import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar #funksiyaning minimal qiymatini hisoblaydi
from scipy.optimize import linprog

# minimumlarni topish
def print_a_function(f, x):
    result = minimize_scalar(f, method='brent')
    print(f'x_min: {result.x}, f(x_min): {result.fun}')
# linspace -> oraligdagi sonlar
    x = np.linspace(result.x -1, result.x+1, 100)
    y = [f(val) for val in x]

    plt.plot(x, y, color='blue', label='f')
    plt.scatter(result.x, result.y, color-'red')
    plt.grid()
    plt.legend(loc=1)

# x ing ildizini -> x0 ni topish
def find_root_bisection(f, min, max):
    epsilon = min

    while ((max-min)/2 >= 0.01):
        epsilon = (max+min)/2
        if f(epsilon) == 0:
            break
        elif (f(epsilon) * f(min) < 0):
            max = epsilon
        else:
            min = epsilon
    return epsilon

# Nyuton_Rawson metodidan foydalanib ildizni topish
def find_root_newton_raphson(f, f_deriv, x_now):
    x_next = x_now - f(x_now)/f_deriv(x_now)
    
    while abs(f(x_next)) > 0.01:
        x_next = x_next - f(x_next)/f_deriv(x_next)
    return x_next

# gradient descent or GD is a optimization agoritmm to find the min of function
# gradient of any function is f_deriv/x_deriv
def gradient_descent(f, f_prime, start, learning_rate = 0.1):
    x_now = start

    for x in range(100):
        x_now = x_now - f_prime(x_now) * learning_rate
    return x_now

f = lambda x: (x-1)**4 + x**2
f_prime = lambda x: 4*((x-1)**3) + 2*x # f_hosila
start = -1
x_min = gradient_descent(f, f_prime, start, 0.01)
f_min = f(x_min)

print(f"xmin: {x_min}, f(x_min): {f_min}")

A = np.array([[2,1], [-4,5], [1,-2]])
b = np.array([10, 8, 3])
c = np.array([-1, -2])

# chiziqli muammoni yechish
def solve_linear_problem(A, b, c):
    res = linprog(c,A,b)
    return round(res.fun), res.x

best_value, best_arg = solve_linear_problem(A, b, c)

print(f"The optimal value is: {best_value} and is reached for x = {best_arg}")
