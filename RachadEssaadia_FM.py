import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# FUNCTION 1 : f(x) = x^3 - 2x + 1
# ============================================================

def f1(x):
    return x**3 - 2*x + 1


# ============================================================
# LAGRANGE - FUNCTION 1
# ============================================================

def P_lagrange_1(x):

    L0 = ((x - 1) * (x - 2)) / 2
    L1 = -x * (x - 2)
    L2 = (x * (x - 1)) / 2

    return 1 * L0 + 0 * L1 + 5 * L2


# ============================================================
# NEWTON - FUNCTION 1
# ============================================================

def P_newton_1(x):

    return 1 - x + 3 * x * (x - 1)


# ============================================================
# HERMITE - FUNCTION 1
# ============================================================

def P_hermite_1(x):

    return x**3 - 2*x + 1


# ============================================================
# FUNCTION 2 : f(x) = e^x
# ============================================================

def f2(x):
    return np.exp(x)


# ============================================================
# LAGRANGE - FUNCTION 2
# ============================================================

def P_lagrange_2(x):

    L0 = ((x - 1) * (x - 2)) / 2
    L1 = -x * (x - 2)
    L2 = (x * (x - 1)) / 2

    return L0 + np.e * L1 + np.exp(2) * L2


# ============================================================
# NEWTON - FUNCTION 2
# ============================================================

def P_newton_2(x):

    f01 = np.e - 1

    f12 = np.exp(2) - np.e

    f012 = (f12 - f01) / 2

    return (
        1
        + f01 * x
        + f012 * x * (x - 1)
    )


# ============================================================
# HERMITE - FUNCTION 2
# ============================================================

def P_hermite_2(x):

    # Hermite interpolation using:
    #
    # H(0) = f(0)
    # H'(0) = f'(0)
    # H(1) = f(1)
    # H'(1) = f'(1)
    # H(2) = f(2)
    # H'(2) = f'(2)

    e = np.e

    return (
        1
        + x
        + ((5 * e**2 - 35) / 4) * x**2
        + ((23 / 2) + 4 * e - 3 * e**2) * x**3
        + ((9 * e**2 / 4) - 4 * e - 23 / 4) * x**4
        + (1 + e - e**2 / 2) * x**5
    )


# ============================================================
# INTERPOLATION POINTS
# ============================================================

x_points = np.array([0, 1, 2])


# ============================================================
# DOMAIN
# ============================================================

x = np.linspace(0, 2, 500)


# ============================================================
# FUNCTION 1 : VALUES
# ============================================================

y_f1 = f1(x)

y_l1 = P_lagrange_1(x)

y_n1 = P_newton_1(x)

y_h1 = P_hermite_1(x)


# ============================================================
# FUNCTION 1 : INTERPOLATION GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    y_f1,
    label="Original function"
)

plt.plot(
    x,
    y_l1,
    ":",
    label="Lagrange"
)

plt.plot(
    x,
    y_n1,
    "-.",
    label="Newton"
)

plt.plot(
    x,
    y_h1,
    "--",
    label="Hermite"
)

plt.scatter(
    x_points,
    f1(x_points),
    s=60,
    label="Interpolation points"
)

plt.xlabel("x")
plt.ylabel("y")

plt.title(
    r"Comparison of interpolation methods - $f(x)=x^3-2x+1$"
)

plt.grid(True)

plt.legend()

plt.savefig(
    "fonction1_comparaison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# FUNCTION 1 : ERRORS
# ============================================================

erreur_l1 = np.abs(y_f1 - y_l1)

erreur_n1 = np.abs(y_f1 - y_n1)

erreur_h1 = np.abs(y_f1 - y_h1)


# ============================================================
# FUNCTION 1 : ERROR GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    erreur_l1,
    label="Lagrange error"
)

plt.plot(
    x,
    erreur_n1,
    label="Newton error"
)

plt.plot(
    x,
    erreur_h1,
    label="Hermite error"
)

plt.xlabel("x")

plt.ylabel("Absolute error")

plt.title(
    r"Interpolation errors - $f(x)=x^3-2x+1$"
)

plt.grid(True)

plt.legend()

plt.savefig(
    "fonction1_erreurs.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# FUNCTION 2 : VALUES
# ============================================================

y_f2 = f2(x)

y_l2 = P_lagrange_2(x)

y_n2 = P_newton_2(x)

y_h2 = P_hermite_2(x)


# ============================================================
# FUNCTION 2 : INTERPOLATION GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    y_f2,
    label="Original function"
)

plt.plot(
    x,
    y_l2,
    ":",
    label="Lagrange"
)

plt.plot(
    x,
    y_n2,
    "-.",
    label="Newton"
)

plt.plot(
    x,
    y_h2,
    "--",
    label="Hermite"
)

plt.scatter(
    x_points,
    f2(x_points),
    s=60,
    label="Interpolation points"
)

plt.xlabel("x")

plt.ylabel("y")

plt.title(
    r"Comparison of interpolation methods - $f(x)=e^x$"
)

plt.grid(True)

plt.legend()

plt.savefig(
    "fonction2_comparaison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# FUNCTION 2 : ERRORS
# ============================================================

erreur_l2 = np.abs(y_f2 - y_l2)

erreur_n2 = np.abs(y_f2 - y_n2)

erreur_h2 = np.abs(y_f2 - y_h2)


# ============================================================
# FUNCTION 2 : ERROR GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    x,
    erreur_l2,
    label="Lagrange error"
)

plt.plot(
    x,
    erreur_n2,
    label="Newton error"
)

plt.plot(
    x,
    erreur_h2,
    label="Hermite error"
)

plt.xlabel("x")

plt.ylabel("Absolute error")

plt.title(
    r"Interpolation errors - $f(x)=e^x$"
)

plt.grid(True)

plt.legend()

plt.savefig(
    "fonction2_erreurs.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()


# ============================================================
# MAXIMUM ERRORS
# ============================================================

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)

print()
print("FUNCTION 1 : f(x) = x^3 - 2x + 1")
print("-" * 60)

print(
    "Maximum Lagrange error :",
    np.max(erreur_l1)
)

print(
    "Maximum Newton error   :",
    np.max(erreur_n1)
)

print(
    "Maximum Hermite error  :",
    np.max(erreur_h1)
)


print()
print("FUNCTION 2 : f(x) = e^x")
print("-" * 60)

print(
    "Maximum Lagrange error :",
    np.max(erreur_l2)
)

print(
    "Maximum Newton error   :",
    np.max(erreur_n2)
)

print(
    "Maximum Hermite error  :",
    np.max(erreur_h2)
)


print()
print("=" * 60)
print("FIGURES SAVED")
print("=" * 60)

print("1. fonction1_comparaison.png")
print("2. fonction1_erreurs.png")
print("3. fonction2_comparaison.png")
print("4. fonction2_erreurs.png")

print("=" * 60)
