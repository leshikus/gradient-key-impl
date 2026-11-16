#!/usr/bin/env python3
"""Regenerate the rigorous constants of the gradient-key article in data/gradient-bound/.

    .venv/bin/python scripts/gradient_bound_data.py constants   # interval computations (about an hour)
    .venv/bin/python scripts/gradient_bound_data.py tex         # constants.tex for article.tex

    .venv/bin/python scripts/gradient_bound_data.py exact <threads>   # exact.csv, exact_formula.csv
    .venv/bin/python scripts/gradient_bound_data.py tplus       # tplus.csv (engine gradient_tplus)

Every value is exact; nothing is sampled.
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "python"))

import mpmath  # noqa: E402
from mpmath import iv, libmp  # noqa: E402

from gradient_key.gradient_bound import (  # noqa: E402
    best_tail_alpha,
    chain_upper,
    complete_cost,
    crt_combine,
    cylinder_tail,
    kappa_column,
    rho_upper,
    tail_mu,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "gradient-bound"
S0 = 4000  # blocked key: interval recursion up to S0, tail lemma beyond
S1 = 500  # gradient key: worst-case recursion up to S1, cylinders beyond
X = 1


def constants() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    alpha = best_tail_alpha(S0)
    mu = tail_mu(S0, alpha)
    with open(DATA / "blocked_constants.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["b", "S0", "max_r_upper", "argmax", "alpha", "mu_upper", "c_b_upper"])
        for b in (2, 3, 4):
            u = chain_upper(b, S0)
            best = max(range(1, S0 + 1), key=lambda s: u[s])
            w.writerow([b, S0, up(u[best], 8), best, alpha, up(mu, 8), up(exact(u[best]) + exact(mu), 8)])
            f.flush()
    tail = cylinder_tail(S1, X)
    with open(DATA / "gradient_constants.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "s1", "x", "max_rho_upper", "argmax", "cylinder_tail_upper", "c_K_upper"])
        for K in (20, 30, 40, 60):
            kap = kappa_column(K, S1)
            u = rho_upper(kap, S1)
            best = max(range(1, S1 + 1), key=lambda s: u[s])
            w.writerow([K, S1, X, up(u[best], 8), best, up(tail, 8), up(exact(u[best]) + exact(tail), 8)])
            f.flush()


ENGINE = Path(__file__).resolve().parents[1] / "build" / "cpp" / "gradient_bound"
TUPLE_CAP = 10**9


def partitions_table(size: int) -> list[int]:
    p = [1] + [0] * size
    for k in range(1, size + 1):
        for i in range(k, size + 1):
            p[i] += p[i - k]
    return p


def tuple_count(n: int, m: int) -> int:
    """Number of (chain, integer partitions per cell) pairs the engine visits."""
    p = partitions_table(m)

    @lru_cache(None)
    def go(j: int, s: int) -> int:
        total = p[s]
        for nxt in range(2, s):
            a, q, b = s - nxt, min(s - nxt, nxt), n - j - 1
            if n - j > 0 and q * (n - j) >= s - 1 and (b >= 40 or (a <= 2**b and nxt <= 2**b)):
                total += p[a] * go(j + 1, nxt)
        return total

    return go(0, m)


def formula_grid() -> list[tuple[int, int]]:
    """The cells of exact_formula.csv, cheapest first."""
    cells = set()
    for m in range(2, 33):
        lo = (m - 1).bit_length()
        hi = 24 if m <= 16 else math.ceil(2 * math.log2(m))
        for n in range(max(lo, 1), hi + 1):
            if tuple_count(n, m) <= TUPLE_CAP or (n, m) == (7, 32):
                cells.add((n, m))
    for m in range(4, 13):  # the range of the non-asymptotic form of the fixed-m theorem
        cells.add((math.isqrt(2 * m**3 - 1) + 1, m))  # ceil(sqrt(2) m^(3/2)), in integers
    for n in range(4, 13):  # Result (2) of the article: every m > 12 with m <= min(32, 2^n), for n <= 12
        for m in range(13, min(32, 2**n) + 1):
            cells.add((n, m))
    return sorted(cells, key=lambda c: (tuple_count(*c) * c[0], c))


def combine(line: str) -> tuple[int, int, Fraction]:
    """Chinese remainders for one engine line, with the spare prime as a check (strict: raises ValueError)."""
    n, m, value = crt_combine(line)
    tables = 1
    for i in range(m):
        tables *= 2**n - i
    return n, m, Fraction(value, tables)


def exact(threads: str) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run([str(ENGINE), *map(str, a)], capture_output=True, text=True, check=True).stdout
    path = DATA / "exact.csv"
    if not path.exists():
        with open(path, "w") as f:
            f.write("n,m,numerator,denominator\n")
            for n, top in ((1, 2), (2, 4), (3, 8), (4, 16), (5, 12), (6, 8), (7, 5)):
                for m in range(2, top + 1):
                    f.write(run("exact", n, m))
    path = DATA / "exact_formula.csv"
    done = set()
    if path.exists():
        done = {(int(r["n"]), int(r["m"])) for r in csv.DictReader(open(path))}
    else:
        path.write_text("n,m,numerator,denominator\n")
    for n, m in formula_grid():
        if (n, m) in done:
            continue
        _, _, value = combine(run("formula", n, m, threads))
        with open(path, "a") as f:
            f.write(f"{n},{m},{value.numerator},{value.denominator}\n")


def exact(x) -> Fraction:
    """An exact rational from a Fraction, an int, a decimal string, or an mpmath number (exactly, no rounding)."""
    if isinstance(x, float):
        raise TypeError("a float may already be rounded below the bound; pass an exact value")
    if isinstance(x, (Fraction, int, str)):
        return Fraction(x)
    return Fraction(*libmp.to_rational(mpmath.mpf(x)._mpf_ if not hasattr(x, "_mpf_") else x._mpf_))


def up(v, digits: int) -> str:
    """The least decimal with ``digits`` places that is >= v, for an exact v (see :func:`exact`).

        >>> up(Fraction(10000000000005, 10**13), 3), up("1.000", 3), up(Fraction(-1, 3), 2)
        ('1.001', '1.000', '-0.33')
    """
    q = math.ceil(exact(v) * 10**digits)
    sign = "-" if q < 0 else ""
    s = str(abs(q)).rjust(digits + 1, "0")
    return f"{sign}{s[:-digits]}.{s[-digits:]}" if digits else f"{sign}{s}"


def log2_lower(m: int) -> Fraction:
    """An exact lower bound of log2 m (exact for powers of two)."""
    if m & (m - 1) == 0:
        return Fraction(m.bit_length() - 1)
    iv.prec = 120
    return Fraction(*libmp.to_rational((iv.log(iv.mpf(m)) / iv.log(2))._mpi_[0]))


def tex() -> None:
    art = Path(os.environ.get("ARTICLE_DIR", Path(__file__).resolve().parents[1].parent / "gradient-key")) 
    bc = {int(r["b"]): r for r in csv.DictReader(open(DATA / "blocked_constants.csv"))}
    gc = {int(r["K"]): r for r in csv.DictReader(open(DATA / "gradient_constants.csv"))}
    words = {20: "twenty", 30: "thirty", 40: "forty", 60: "sixty"}
    lines = ["% Generated by scripts/gradient_bound_data.py tex; do not edit."]
    for b, name in ((2, "two"), (3, "three"), (4, "four")):
        lines.append(rf"\newcommand{{\cb{name}}}{{{up(bc[b]['c_b_upper'], 3)}}}")
    for b, name in ((2, "two"), (3, "three")):
        val = (Fraction(bc[b]["c_b_upper"]) + Fraction(1, 64)) / (1 - Fraction(1, 2**13))
        lines.append(rf"\newcommand{{\cor{name}}}{{{up(val, 3)}}}")
    lines.append(rf"\newcommand{{\talpha}}{{{bc[2]['alpha']}}}")
    lines.append(rf"\newcommand{{\tmu}}{{{up(bc[2]['mu_upper'], 4)}}}")
    for K, name in words.items():
        lines.append(rf"\newcommand{{\cG{name}}}{{{up(gc[K]['c_K_upper'], 3)}}}")
    # all m >= 4 (m <= 3 is trivial): the error term is at most max(0.026/16, 0.026/500^2 + 2^-10) / P(D)
    e = Fraction(26, 1000)
    full = Fraction(gc[30]["c_K_upper"]) + max(e / 16, e / 500**2 + Fraction(1, 2**10)) / (1 - Fraction(1, 2 * 4**28))
    lines.append(rf"\newcommand{{\cGthirtyfull}}{{{up(full, 3)}}}")
    lines.append(rf"\newcommand{{\cyltail}}{{{up(gc[30]['cylinder_tail_upper'], 4)}}}")
    t = {}
    for r in csv.DictReader(open(DATA / "exact_formula.csv")):
        t[int(r["n"]), int(r["m"])] = Fraction(int(r["numerator"]), int(r["denominator"]))
    red = {k: v - log2_lower(k[1]) for k, v in t.items()}  # exact upper bounds of t - log m
    worst = max(red, key=red.get)
    lines.append(rf"\newcommand{{\exactcells}}{{{len(t)}}}")
    lines.append(rf"\newcommand{{\exactcellsbig}}{{{sum(1 for n, m in t if n <= 12 and m > 12)}}}")  # result (2)
    lines.append(rf"\newcommand{{\exactmaxred}}{{{up(red[worst], 4)}}}")
    lines.append(rf"\newcommand{{\exactmaxat}}{{({worst[0]},{worst[1]})}}")
    tops = {}
    for m in range(2, 33):
        n = max((m - 1).bit_length(), 1)
        if (n, m) not in t:
            continue
        while (n + 1, m) in t:
            n += 1
        tops[m] = n
    groups: list[list[int]] = []
    for m, n in tops.items():
        if groups and groups[-1][2] == n and groups[-1][1] == m - 1:
            groups[-1][1] = m
        else:
            groups.append([m, m, n])
    text = "; ".join(
        (f"$m={a}$" if a == b else f"$m={a},\\dots,{b}$") + f", $\\lceil\\log m\\rceil\\le n\\le{n}$" for a, b, n in groups
    )
    lines.append(rf"\newcommand{{\exactrange}}{{{text}}}")
    (art / "constants.tex").write_text("\n".join(lines) + "\n")


TPLUS_ENGINE = Path(__file__).resolve().parents[1] / "build" / "cpp" / "gradient_tplus"
def tplus() -> None:
    subprocess.run([str(TPLUS_ENGINE), "grid", str(DATA / "tplus.csv"), "--data", str(DATA)], check=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["constants"]:
        constants()
    elif sys.argv[1:] == ["tex"]:
        tex()
    elif sys.argv[1:] == ["tplus"]:
        tplus()
    elif sys.argv[1:2] == ["exact"] and len(sys.argv) == 3:
        exact(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
