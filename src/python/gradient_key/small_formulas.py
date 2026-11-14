r"""
Exact formulas for the average cost of the gradient key when the number of columns `m` is small.

Let `M = 2^{m-1}` be the number of row types: the patterns in `{0,1}^m` up to complement
(the constant row is one type). The gradient key `G` depends on a table only through the order
in which the types first appear in the rows, and a table has `2^n` orientations of one type
sequence. A sequence of `n` rows that uses exactly `k` types in a given order of first appearance
can be chosen in `S(n,k)` ways (Stirling numbers of the second kind). So, with `B_k` the sum of
`Cost - C(m)` over ordered `k`-tuples of distinct types that separate the columns,

    N(n) = sum over tables with distinct columns of (Cost - C(m)) = 2^n sum_k B_k S(n,k)
         = sum_{j=1}^{M} a_j (2j)^n,        a_j = sum_{k>=j} B_k (-1)^{k-j} C(k,j) / k!,

and `t_{n,m}(G) - C(m) = N(n) / (2^n)_m` with `(2^n)_m = 2^n (2^n-1) ... (2^n-m+1)`.
`N(n) = 0` for `n < ceil(log2 m)`, as no table has distinct columns. The `M` values
`N(0), ..., N(M-1)` therefore determine the `M` coefficients `a_j` exactly (Vandermonde system),
and further values of `n` check them. The leading term is `a_J (J/M)^n`, with `J` the largest `j`
with `a_j != 0`.

EXAMPLES::

    >>> direct_coefficients(3)
    {}
    >>> c = direct_coefficients(4)
    >>> excess(c, 4, 3)
    Fraction(1, 35)
    >>> leading_term(c, 4)
    (5, Fraction(1, 4), Fraction(5, 8))
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import itertools
import json
import subprocess
from fractions import Fraction
from math import comb, factorial
from pathlib import Path

from gradient_key.gradient_bound import complete_cost


def falling(x: int, m: int) -> int:
    """`x (x-1) ... (x-m+1)`."""
    out = 1
    for i in range(m):
        out *= x - i
    return out


def tree_cost_sum(columns: int, rows: list[int]) -> int | None:
    """Sum of leaf depths of the gradient tree on `columns` columns (bit masks of the rows), or None if not separating."""

    def go(s: int) -> int | None:
        size = s.bit_count()
        if size < 2:
            return 0
        best, pick = 0, 0
        for r in rows:
            k = (s & r).bit_count()
            q = min(k, size - k)
            if q > best:
                best, pick = q, r
        if best == 0:
            return None
        left, right = go(s & pick), go(s & ~pick)
        return None if left is None or right is None else size + left + right

    return go((1 << columns) - 1)


def types(m: int) -> list[int]:
    """The `2^(m-1)` row types as bit masks of columns, column 0 fixed to 0."""
    return [r << 1 for r in range(2 ** (m - 1))]


def direct_coefficients(m: int) -> dict[int, Fraction]:
    """The `a_j` from the ordered tuples of types; practical for `m <= 4` only (it enumerates `M!` tuples)."""
    assert 2 <= m <= 4, f"got [{m}]"
    ts = types(m)
    c_total = complete_cost(m) * m
    b: dict[int, Fraction] = {}
    for k in range(1, len(ts) + 1):
        total = Fraction(0)
        for tup in itertools.permutations(ts, k):
            s = tree_cost_sum(m, list(tup))
            if s is not None:
                total += s - c_total
        if total:
            b[k] = total / m
    a: dict[int, Fraction] = {}
    for j in range(1, len(ts) + 1):
        v = sum(b.get(k, 0) * (-1) ** (k - j) * Fraction(comb(k, j), factorial(k)) for k in range(j, len(ts) + 1))
        if v:
            a[j] = Fraction(v)
    return a


def numerator(coeffs: dict[int, Fraction], n: int) -> Fraction:
    """`N(n) = sum_j a_j (2j)^n`."""
    return sum((a * (2 * j) ** n for j, a in coeffs.items()), Fraction(0))


def excess(coeffs: dict[int, Fraction], m: int, n: int) -> Fraction:
    """`t_{n,m}(G) - C(m)` from the coefficients; needs `m <= 2^n`."""
    assert 2 <= m <= 2**n, f"got n={n}, m={m}"
    return numerator(coeffs, n) / falling(2**n, m)


def fit_coefficients(values: list[Fraction], m: int) -> dict[int, Fraction]:
    """Solve for `a_1..a_M` from `N(0), ..., N(M-1)` by Lagrange interpolation on the nodes `2j`."""
    big_m = 2 ** (m - 1)
    assert len(values) >= big_m, f"need {big_m} values, got [{len(values)}]"
    master = [1]
    for j in range(1, big_m + 1):
        nxt = [0] * (len(master) + 1)
        for i, c in enumerate(master):
            nxt[i + 1] += c
            nxt[i] -= 2 * j * c
        master = nxt
    out: dict[int, Fraction] = {}
    for j in range(1, big_m + 1):
        q, carry = [0] * big_m, 0
        for i in range(big_m, 0, -1):
            carry = master[i] + 2 * j * carry
            q[i - 1] = carry
        den = 1
        for l in range(1, big_m + 1):
            if l != j:
                den *= 2 * (j - l)
        v = sum(Fraction(q[i]) * values[i] for i in range(big_m)) / den
        if v:
            out[j] = v
    return out


def leading_term(coeffs: dict[int, Fraction], m: int) -> tuple[int, Fraction, Fraction]:
    """`(J, a_J, J/M)`: the excess is `a_J (J/M)^n (1 + o(1))` as `n -> infinity`."""
    j = max(coeffs)
    return j, coeffs[j], Fraction(j, 2 ** (m - 1))


def engine_excess_numerator(n: int, m: int, engine: Path, threads: int = 8) -> Fraction:
    """`N(n)` from the exact path-formula engine (residues modulo 62-bit primes, combined by Chinese remainders)."""
    if n < (m - 1).bit_length():
        return Fraction(0)
    line = subprocess.run([str(engine), "formula", str(n), str(m), str(threads)], capture_output=True, text=True, check=True).stdout
    pairs = [tuple(map(int, x.split(":"))) for x in line.strip().split(",")[4:]]
    value, modulus = 0, 1
    for prime, residue in pairs[:-1]:
        value += modulus * (((residue - value) * pow(modulus, -1, prime)) % prime)
        modulus *= prime
    assert value % pairs[-1][0] == pairs[-1][1], f"spare prime disagrees at n={n}, m={m}"
    tables = falling(2**n, m)
    assert tables <= value <= (m - 1) * tables < modulus, f"reconstruction out of range at n={n}, m={m}"
    return (Fraction(value, tables) - complete_cost(m)) * tables


def formula_for(m: int, engine: Path, extra: int = 8) -> dict:
    """Coefficients for `m`, fitted on `n < M` and checked on `M <= n < M + extra`."""
    big_m = 2 ** (m - 1)
    values = [engine_excess_numerator(n, m, engine) for n in range(big_m + extra)]
    coeffs = fit_coefficients(values[:big_m], m)
    for n in range(big_m, big_m + extra):
        assert numerator(coeffs, n) == values[n], f"check fails at n={n}, m={m}"
    j, a, r = leading_term(coeffs, m) if coeffs else (0, Fraction(0), Fraction(0))
    return {"m": m, "types": big_m, "checked_to_n": big_m + extra - 1, "coefficients": {str(k): str(v) for k, v in coeffs.items()},
            "leading": {"J": j, "A": str(a), "rate": str(r)}}


def load_formula(data_dir: Path, m: int) -> dict[int, Fraction]:
    """Read the coefficients of `m` from `formula_m{m}.json`."""
    raw = json.loads((data_dir / f"formula_m{m}.json").read_text())
    return {int(k): Fraction(v) for k, v in raw["coefficients"].items()}


def frac_tex(x: Fraction, cmd: str = "tfrac") -> str:
    """`p/q` as TeX, with a sign in front."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    return sign + (str(x.numerator) if x.denominator == 1 else rf"\{cmd}{{{x.numerator}}}{{{x.denominator}}}")


def formula_tex(m: int, coeffs: dict[int, Fraction]) -> str:
    """`N(n)` for the columns count `m` as a TeX display."""
    terms = []
    for j, a in sorted(coeffs.items()):
        sign = "-" if a < 0 else "+"
        mag = abs(a)
        coef = "" if mag == 1 else frac_tex(mag) + r"\,"
        terms.append(f" {sign} {coef}{2 * j}^n")
    body = "".join(terms).lstrip()
    body = body[2:] if body.startswith("+ ") else "-" + body[2:]
    return rf"N_{{{m}}}(n)={body}"


def table_m_tex(data_dir: Path, ms: range) -> str:
    """Table of the leading terms for few columns."""
    rows = [r"\begin{tabular}{rrrrrr}", r"\toprule",
            r"$m$ & terms & $J$ & $A_m$ & $r_m=J/2^{m-1}$ & $r_m$ decimal \\", r"\midrule"]
    for m in ms:
        raw = json.loads((data_dir / f"formula_m{m}.json").read_text())
        lead = raw["leading"]
        if not lead["J"]:
            rows.append(rf"{m} & 0 & & $0$ & & \\")
            continue
        a, r = Fraction(lead["A"]), Fraction(lead["rate"])
        rows.append(rf"{m} & {len(raw['coefficients'])} & {lead['J']} & ${frac_tex(a, "frac")}$ & ${frac_tex(r, "frac")}$ & {float(r):.4f} \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"


def fixed_n_rows(path: Path) -> dict[int, dict[int, Fraction]]:
    """Read `fixed_n.csv` as `{n: {m: excess}}`."""
    import csv

    out: dict[int, dict[int, Fraction]] = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(int(r["n"]), {})[int(r["m"])] = Fraction(int(r["excess_numerator"]), int(r["excess_denominator"]))
    return out


def table_n_tex(path: Path) -> str:
    """Table of the largest excess for few rows."""
    rows = [r"\begin{tabular}{rrrrrr}", r"\toprule",
            r"$n$ & columns $m$ & $m$ with excess $>0$ & largest excess & at $m$ & decimal \\", r"\midrule"]
    for n, cells in sorted(fixed_n_rows(path).items()):
        pos = sum(1 for v in cells.values() if v > 0)
        top_m = max(cells, key=lambda k: cells[k])
        top = cells[top_m]
        rows.append(rf"{n} & $2,\dots,{2**n}$ & {pos} & ${frac_tex(top, "frac")}$ & {top_m if top else ''} & {float(top):.5f} \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(rows) + "\n"
