"""Sanity check (not a proof): per table, Lbar <= l-1+m 2^-l + sum_{i<l} 2^(i-l) V_i(A), and E V_i <= n/(n-i)."""
import random, math, sys
def gtree(A, n, m):
    # returns per-column list of (s_i, w_i) along the path
    rec = {j: [] for j in range(m)}
    stack = [(list(range(m)), 0)]
    while stack:
        S, d = stack.pop()
        s = len(S)
        if s == 1:
            continue
        best, br = -1, None
        for r in range(n):
            k = sum(A[r][c] for c in S)
            q = min(k, s - k)
            if q > best:
                best, br = q, r
        k = sum(A[br][c] for c in S)
        w = (s - 2 * k) ** 2 / s
        for c in S:
            rec[c].append((s, w))
        stack.append(([c for c in S if A[br][c]], d + 1))
        stack.append(([c for c in S if not A[br][c]], d + 1))
    return rec
def table(n, m):
    cols = random.sample(range(2 ** n), m)
    return [[(c >> r) & 1 for c in cols] for r in range(n)]
worst = -1e9
for n, m, T in [(5, 20, 300), (6, 40, 300), (8, 100, 200), (10, 200, 100), (12, 1000, 30), (9, 512, 30), (7, 128, 50)]:
    l = int(math.floor(math.log2(m)))
    Vsum = [0.0] * n
    for _ in range(T):
        A = table(n, m)
        rec = gtree(A, n, m)
        Lbar = sum(len(v) for v in rec.values()) / m
        V = [0.0] * n
        for v in rec.values():
            for i, (s, w) in enumerate(v):
                V[i] += w / m
        rhs = l - 1 + m / 2 ** l + sum(2 ** (i - l) * V[i] for i in range(l))
        worst = max(worst, Lbar - rhs)
        for i in range(n):
            Vsum[i] += V[i] / T
    print(n, m, "max_i E V_i*(n-i)/n =", round(max(Vsum[i] * (n - i) / n for i in range(n)), 3), "| t - log m =", round(Lbar - math.log2(m), 3), flush=True)
print("max over tables of Lbar - rhs:", worst, "(must be <= 0)")
