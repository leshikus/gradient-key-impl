"""Write data/gradient-bound/formula_cells.csv: upper bounds of t_{n,m}(G) from the exact formulas of
Proposition small-m (data/small/formula_m{m}.json), for the cells of the t^+ grid with 2 <= m <= 12
that have no value in exact_formula.csv.

Columns: n, m, t_upper (a decimal upper bound: the exact rational for n <= 4096, the upper end of an
mpmath interval otherwise, rounded up to 25 decimals), method ("exact" or "interval").
The engine gradient_tplus reads the file into ingredient (a).

    .venv/bin/python scripts/formula_cells_data.py
"""
from __future__ import annotations

import csv
import math
from fractions import Fraction
from pathlib import Path

import mpmath
from mpmath import libmp

import gradient_key.gradient_tplus as gt
from gradient_key.small_formulas import load_formula

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gradient-bound"
NS = list(range(1, 13)) + [16, 20, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 4096, 2**16, 2**20]
DIGITS = 25


def frac(x) -> Fraction:
    """An mpmath number, or a degenerate interval such as ``x.a`` or ``x.b``, as an exact Fraction.

    ``mpmath.mpf(x)`` would round an interval endpoint to the working precision of ``mpmath.mp``, so the
    endpoint is read exactly from ``_mpi_``.
    """
    if hasattr(x, "_mpi_"):
        lo, hi = x._mpi_
        assert lo == hi, "pass an endpoint (x.a or x.b), not an interval"
        p, q = libmp.to_rational(hi)
        return Fraction(p, q)
    p, q = libmp.to_rational(mpmath.mpf(x)._mpf_)
    return Fraction(p, q)


def ceil_decimal(x: Fraction) -> str:
    q = math.ceil(x * 10**DIGITS)
    s = str(q).rjust(DIGITS + 1, "0")
    return s[:-DIGITS] + "." + s[-DIGITS:]


def main() -> None:
    with open(DATA / "exact_formula.csv", newline="") as f:
        have = {(int(r["n"]), int(r["m"])) for r in csv.DictReader(f)}
    rows = []
    for m in range(2, 13):
        coeffs = load_formula(ROOT / "data" / "small", m)
        for n in NS:
            if m > 2**n or (n, m) in have:
                continue
            b = gt.formula_value(n, m, coeffs)
            rows.append((n, m, ceil_decimal(frac(b.b)), "exact" if n <= 4096 else "interval"))
    with open(DATA / "formula_cells.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["n", "m", "t_upper", "method"])
        w.writerows(rows)
    print(f"{len(rows)} cells written to {DATA / 'formula_cells.csv'}")


if __name__ == "__main__":
    main()
