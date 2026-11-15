#!/usr/bin/env python3
"""Certify every cell of the t^+ grid in interval arithmetic.

    .venv/bin/python scripts/certify_tplus.py [processes]    # needs build/cpp/gradient_tplus

For each cell of data/gradient-bound/tplus.csv the script recomputes the ingredient that gives the
minimum with mpmath interval arithmetic (gradient_key.gradient_tplus), or takes the exact rational
value, and checks, as exact rationals:

- the stored t^+ is at least the upper end of the certified bound;
- the value printed in Table tplus of the article, t^+ - log m rounded up, is at least that upper
  end minus a lower bound of log m.

The engine is used only for the requirement kappa of ingredient (f); any requirement gives a valid
bound, so it needs no certificate. Exit status 0 means every cell passed.
"""
# SPDX-License-Identifier: MIT
from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "python"))

import mpmath  # noqa: E402
from mpmath import iv, libmp  # noqa: E402

import gradient_key.gradient_tplus as gt  # noqa: E402
from gradient_key.small_formulas import load_formula  # noqa: E402
from gradient_key.gradient_bound import complete_cost  # noqa: E402

DATA = ROOT / "data" / "gradient-bound"
ENGINE = Path(os.environ.get("GRADIENT_KEY_BUILD", ROOT / "build" / "cpp")) / "gradient_tplus"


def _frac(x) -> Fraction:
    """An mpmath number, or an interval endpoint (x.a, x.b), as an exact rational (no rounding to mp.prec)."""
    if hasattr(x, "_mpi_"):
        lo, hi = x._mpi_
        assert lo == hi, "pass an endpoint (x.a or x.b), not an interval"
        return Fraction(*libmp.to_rational(hi))
    return Fraction(*libmp.to_rational(mpmath.mpf(x)._mpf_))


def log2_lower(m: int) -> Fraction:
    if m & (m - 1) == 0:  # exact for powers of two
        return Fraction(m.bit_length() - 1)
    iv.prec = 120
    return _frac((iv.log(iv.mpf(m)) / iv.log(2)).a)


def params(text: str) -> dict:
    return dict(t.split("=") for t in text.replace("(s1=m)", "").split())


def printed(r: dict) -> Fraction:
    """The decimal of Table tplus (scripts/gradient_bound_data.py, tplus_cell)."""
    v = Fraction(r["tplus_minus_log2m"])
    return Fraction(math.ceil(v * 1000), 1000) if v < 10 else Fraction(math.ceil(v * 10), 10)


def certified_upper(r: dict, exact: dict) -> Fraction:
    n, m, ing = int(r["n"]), int(r["m"]), r["ingredient"]
    if ing == "m=1":
        return Fraction(0)
    if ing == "exact":
        if (n, m) in exact:
            return exact[n, m]
        if r["params"] == "dense":  # Proposition dense: t = C(m)
            return complete_cost(m)
        if r["params"] == "formula":  # Proposition small-m, recomputed from the coefficients
            return _frac(gt.formula_value(n, m, load_formula(ROOT / "data" / "small", m)).b)
        raise ValueError(f"exact winner without an exact value at ({n},{m})")
    if ing == "uniform":
        return _frac(gt.uniform_bound(n, m).b)
    if ing == "trivial":
        return Fraction(gt.trivial_bound(n, m))
    if ing == "fixed":
        return _frac(gt.fixed_bound(n, m).b)
    if ing == "union":
        p = params(r["union_params"])
        return _frac(gt.union_bound(n, m, int(p["j"]), int(p["s1"]), float(p["x"])).b)
    if ing == "cylinder":
        p = params(r["cylinder_params"])
        s1 = int(p["s1"])
        out = subprocess.run([str(ENGINE), "kappa", str(n), str(m), p["j"], "--data", str(DATA)],
                             capture_output=True, text=True, check=True).stdout.split()
        kap = ([0, 0] + [int(v) for v in out[1::2]])[: min(s1, m) + 1]
        return _frac(gt.cylinder_bound(n, m, kap, s1, float(p["x"])).b)
    raise ValueError(f"no certificate for ingredient {ing} at ({n},{m})")


def check(args) -> tuple[int, int, str, str]:
    r, exact = args
    n, m = int(r["n"]), int(r["m"])
    t0 = time.time()
    try:
        up = certified_upper(r, exact)
    except Exception as e:  # noqa: BLE001
        return n, m, r["ingredient"], f"FAIL {e}"
    bad = []
    if Fraction(r["tplus"]) < up:
        bad.append("stored t+ below the certified bound")
    if m > 1 and printed(r) < up - log2_lower(m):
        bad.append("printed value below the certified bound")
    if m > 1 and up < complete_cost(m):
        bad.append("certified bound below C(m)")
    return n, m, r["ingredient"], ("FAIL " + "; ".join(bad)) if bad else f"ok {time.time() - t0:.1f}s"


def main() -> int:
    procs = int(sys.argv[1]) if len(sys.argv) > 1 else os.cpu_count() or 1
    rows = list(csv.DictReader(open(DATA / "tplus.csv")))
    exact = {(int(r["n"]), int(r["m"])): Fraction(int(r["numerator"]), int(r["denominator"]))
             for r in csv.DictReader(open(DATA / "exact_formula.csv"))}
    # slow cells first, for load balance
    rows.sort(key=lambda r: (r["ingredient"] != "cylinder", -int(r["m"])))
    fails = 0
    with ProcessPoolExecutor(procs) as ex:
        for n, m, ing, status in ex.map(check, [(r, exact) for r in rows], chunksize=1):
            if status.startswith("FAIL"):
                fails += 1
            print(f"{n} {m} {ing} {status}", flush=True)
    print(f"\n{len(rows)} cells, {fails} failure(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
