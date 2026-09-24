import sympy as sp

x = sp.symbols('x')
n = 4
X = sp.symbols(f'x0:{n+1}')      # x0 ... x4
F = sp.symbols(f'f0:{n+1}')      # f0 ... f4  (valeurs f(x_k))

def L(k, nodes):
    Lk = sp.Integer(1)
    for j in range(len(nodes)):
        if j != k:
            Lk *= (x - nodes[j]) / (nodes[k] - nodes[j])
    return Lk

#Partie générale
print("Formule générale (n = 4)")
for k in range(n + 1):           # L_0 ... L_4
    print(f"L_{k}(x) =", L(k, X))

P4 = sum(F[k] * L(k, X) for k in range(n + 1))
print("\nP_4(x) =", P4)

#Application aux points (1,3), (10,7), (8,5)
print("\n Points (1,3), (10,7), (8,5)")
xs = [1, 10, 8]
ys = [3, 7, 5]
for k in range(len(xs)):
    print(f"L_{k}(x) =", sp.expand(L(k, xs)))

P = sp.expand(sum(y * L(k, xs) for k, y in enumerate(ys)))
print("P(x) =", P)