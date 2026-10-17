import time

def gauss_swap(A, b):
    """Gauss with row swapping only when the pivot is zero."""
    n = len(b); A = [r[:] for r in A]; b = b[:]
    for k in range(n - 1):
        if A[k][k] == 0:
            for i in range(k + 1, n):
                if A[i][k] != 0:
                    A[k], A[i] = A[i], A[k]; b[k], b[i] = b[i], b[k]; break
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n): A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
    return back(A, b)

def gauss_partial(A, b):
    """Gauss with partial pivoting (largest |a_ik| in column k)."""
    n = len(b); A = [r[:] for r in A]; b = b[:]
    for k in range(n - 1):
        p = max(range(k, n), key=lambda i: abs(A[i][k]))
        A[k], A[p] = A[p], A[k]; b[k], b[p] = b[p], b[k]
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n): A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
    return back(A, b)

def gauss_total(A, b):
    """Gauss with total pivoting (largest |a_ij| in the remaining submatrix)."""
    n = len(b); A = [r[:] for r in A]; b = b[:]
    perm = list(range(n))            # tracks column swaps
    for k in range(n - 1):
        pi, pj, best = k, k, -1
        for i in range(k, n):
            for j in range(k, n):
                if abs(A[i][j]) > best: best, pi, pj = abs(A[i][j]), i, j
        A[k], A[pi] = A[pi], A[k]; b[k], b[pi] = b[pi], b[k]
        for r in A: r[k], r[pj] = r[pj], r[k]
        perm[k], perm[pj] = perm[pj], perm[k]
        for i in range(k + 1, n):
            m = A[i][k] / A[k][k]
            for j in range(k, n): A[i][j] -= m * A[k][j]
            b[i] -= m * b[k]
    y = back(A, b)
    x = [0.0] * n
    for k in range(n): x[perm[k]] = y[k]
    return x

def back(A, b):
    n = len(b); x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - sum(A[i][j] * x[j] for j in range(i + 1, n))) / A[i][i]
    return x

if __name__ == "__main__":
    A = [[1, 2, 3], [4, 5, 6], [7, 8, 10]]; b = [14, 32, 53]
    N = 20000
    for f in (gauss_swap, gauss_partial, gauss_total):
        t = time.process_time()
        for _ in range(N): x = f(A, b)
        print(f"{f.__name__:15s} x={[round(v,6) for v in x]}  CPU = {time.process_time()-t:.4f} s")
