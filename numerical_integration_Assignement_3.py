
import math


def f(x):
    """Integrand: f(x) = exp(-x^2)."""
    return math.exp(-x**2)


def composite_trapezoidal(a, b, n):
    """Compute the composite trapezoidal approximation."""
    h = (b - a) / n

    total = (f(a) + f(b)) / 2

    for i in range(1, n):
        x = a + i * h
        total += f(x)

    return h * total


def composite_simpson(a, b, n):
    """Compute the composite Simpson approximation."""
    if n % 2 != 0:
        raise ValueError("Simpson's method requires an even n.")

    h = (b - a) / n
    total = f(a) + f(b)

    for i in range(1, n):
        x = a + i * h

        if i % 2 == 1:
            total += 4 * f(x)
        else:
            total += 2 * f(x)

    return (h / 3) * total


def main():
    a = 0.0
    b = 1.0
    values_n = [4, 6]

    # Reference value: integral from 0 to 1 of exp(-x^2)
    reference = math.sqrt(math.pi) / 2 * math.erf(1)

    print("Integral: I = integral from 0 to 1 of exp(-x^2)")
    print(f"Reference value: {reference:.10f}\n")

    print(
        f"{'N':<5}"
        f"{'Trapezoidal':<18}"
        f"{'Simpson':<18}"
    )
    print("-" * 41)

    for n in values_n:
        trapezoidal = composite_trapezoidal(a, b, n)
        simpson = composite_simpson(a, b, n)

        error_trapezoidal = abs(reference - trapezoidal)
        error_simpson = abs(reference - simpson)

        print(
            f"{n:<5}"
            f"{trapezoidal:<18.10f}"
            f"{simpson:<18.10f}"
        )

        print(f"     Trapezoidal absolute error: "
              f"{error_trapezoidal:.10f}")

        print(f"     Simpson absolute error:     "
              f"{error_simpson:.10f}\n")


if __name__ == "__main__":
    main()
