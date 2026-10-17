import time


def show(A, b, title, names=None):
    """Print the augmented matrix [A | b] (and the current unknown order)."""
    print(title + (f"   unknowns: {names}" if names else ""))
    for row, bi in zip(A, b):
        print("   " + "  ".join(f"{v:8.4f}" for v in row) + "  |  " + f"{bi:8.4f}")

def back_substitution(A, b):
    """Solve the upper triangular system by going from the last row to the first."""
    n = len(b)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / A[i][i]
    return x

def eliminate(A, b, k):
    """Cancel column k below the pivot A[k][k]:  L_i <- L_i - m_ik * L_k."""
    n = len(b)
    for i in range(k + 1, n):
        m = A[i][k] / A[k][k]
        for j in range(k, n):
            A[i][j] -= m * A[k][j]
        b[i] -= m * b[k]

def gauss_swap(A, b, verbose=False):
    """Gauss with row permutation: swap rows ONLY if the pivot is 0."""
    n = len(b)
    A = [r[:] for r in A]; b = b[:]
    for k in range(n - 1):
        if A[k][k] == 0:
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    A[k], A[i] = A[i], A[k]
                    b[k], b[i] = b[i], b[k]
                    break
            else:
                raise ValueError("singular matrix")
        eliminate(A, b, k)
        if verbose: show(A, b, f"after step {k+1}")
    return back_substitution(A, b)

def gauss_partial(A, b, verbose=False):
    """Gauss with partial pivoting: largest |a_ik| in column k (rows k..n-1)."""
    n = len(b)
    A = [r[:] for r in A]; b = b[:]
    for k in range(n - 1):
        p = max(range(k, n), key=lambda i: abs(A[i][k]))
        A[k], A[p] = A[p], A[k]
        b[k], b[p] = b[p], b[k]
        eliminate(A, b, k)
        if verbose: show(A, b, f"after step {k+1} (row swap {k+1}<->{p+1})")
    return back_substitution(A, b)

def gauss_total(A, b, verbose=False):
    """Gauss with total pivoting: largest |a_ij| in the whole submatrix.
    Column swaps swap the unknowns, so we record them in `names` and undo them at the end."""
    n = len(b)
    A = [r[:] for r in A]; b = b[:]
    names = list(range(n))                     
    for k in range(n - 1):
        # 1) search the pivot in the submatrix rows k.., columns k..
        p, q, best = k, k, -1.0
        for i in range(k, n):
            for j in range(k, n):
                if abs(A[i][j]) > best:
                    best, p, q = abs(A[i][j]), i, j
        # 2) swap rows k <-> p
        A[k], A[p] = A[p], A[k]
        b[k], b[p] = b[p], b[k]
        # 3) swap columns k <-> q (ALL rows) and record it
        for row in A:
            row[k], row[q] = row[q], row[k]
        names[k], names[q] = names[q], names[k]
        # 4) eliminate
        eliminate(A, b, k)
        if verbose:
            show(A, b, f"after step {k+1} (rows {k+1}<->{p+1}, cols {k+1}<->{q+1})",
                 [ "xyz"[c] if n == 3 else f"x{c+1}" for c in names])
    y = back_substitution(A, b)
    x = [0.0] * n
    for j in range(n):
        x[names[j]] = y[j]                    
    return x


if __name__ == "__main__":
    A = [[0, 1, 1],
         [2, 1, 1],
         [1, 2, 3]]
    b = [5, 7, 14]

    for f in (gauss_swap, gauss_partial, gauss_total):
        print("=" * 60); print(f.__name__)
        x = f(A, b, verbose=True)
        print("solution (x, y, z) =", [round(v, 10) for v in x])

    h = 20000
    print("=" * 60); print(f"CPU time for h = {h} solves of the common example")
    for f in (gauss_swap, gauss_partial, gauss_total):
        t = time.process_time()
        for _ in range(h):
            f(A, b)
        print(f"{f.__name__:15s} {time.process_time() - t:.4f} s")
