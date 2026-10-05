import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**3 - 2*x + 1





# 2. Polynôme de Lagrange


def P_lagrange(x):
    L0 = ((x - 1) * (x - 2)) / 2
    L1 = -x * (x - 2)
    L2 = (x * (x - 1)) / 2

    return 1*L0 + 0*L1 + 5*L2


# 3. Polynôme de Newton


def P_newton(x):
    return 1 - x + 3*x*(x - 1)



# 4. Polynôme de Hermite


def P_hermite(x):
    return x**3 - 2*x + 1




x_points = np.array([0, 1, 2])
y_points = f(x_points)



x = np.linspace(0, 2, 500)




y_f = f(x)
y_l = P_lagrange(x)
y_n = P_newton(x)
y_h = P_hermite(x)



plt.figure(figsize=(10, 6))

plt.plot(x, y_f, label="Fonction f(x)")
plt.plot(x, y_l, ":", label="Lagrange")
plt.plot(x, y_n, "-.", label="Newton")
plt.plot(x, y_h, label="Hermite")

plt.scatter(
    x_points,
    y_points,
    s=60,
    label="Points d'interpolation"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparaison des méthodes d'interpolation")

plt.grid(True)
plt.legend()
plt.show()