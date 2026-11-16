"""The missing-set engine build/cpp/gradient_complement against independent exact values.

The engine enumerates the r = 2^n - m missing words (see src/cpp/gradient_complement.cpp). It must agree,
as exact fractions, with the brute-force enumeration (exact.csv), the path formula (exact_formula.csv) and the closed
forms of Propositions dense and near-dense. Stored values of exact_complement.csv must agree with every other source.
"""
from __future__ import annotations

import csv
import os
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from gradient_key.gradient_bound import complete_cost
from gradient_key.small_formulas import closed_form_excess

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "gradient-bound"
ENGINE = Path(os.environ.get("GRADIENT_KEY_BUILD", ROOT / "build" / "cpp")) / "gradient_complement"


def rows(name):
    path = DATA / name
    if not path.exists():
        return {}
    return {(int(r["n"]), int(r["m"])): Fraction(int(r["numerator"]), int(r["denominator"])) for r in csv.DictReader(open(path))}


def references(n, m):
    out = []
    for name in ("exact.csv", "exact_formula.csv"):
        v = rows(name).get((n, m))
        if v is not None:
            out.append(v)
    cf = closed_form_excess(n, m)
    if cf is not None:
        out.append(complete_cost(m) + cf[0])
    return out


def engine(n, m):
    if not ENGINE.exists():
        pytest.skip("build the engines first (cmake --build build/cpp)")
    line = subprocess.run([str(ENGINE), str(n), str(2**n - m), "4"], capture_output=True, text=True, check=True).stdout
    nn, mm, num, den = line.strip().split(",")
    assert (int(nn), int(mm)) == (n, m)
    return Fraction(int(num), int(den))


# every m for n <= 4; for n = 5 the cells with few or many missing words (the middle ones take seconds each);
# for n = 6 the closed forms r <= 8 except the slowest
CELLS = [(n, m) for n in (2, 3, 4) for m in range(2, 2**n)]
CELLS += [(5, m) for m in list(range(2, 8)) + list(range(24, 32))]
CELLS += [(6, m) for m in range(57, 64)]


@pytest.mark.parametrize("cell", CELLS)
def test_engine_agrees_with_independent_values(cell):
    refs = references(*cell)
    assert refs, cell
    v = engine(*cell)
    assert all(v == r for r in refs), (cell, v, refs)


def test_engine_rejects_bad_arguments():
    if not ENGINE.exists():
        pytest.skip("build the engines first")
    for args in (["7", "3", "1"], ["6", "0", "1"], ["6", "63", "1"], ["6", "9x", "1"], ["6", "9"]):
        assert subprocess.run([str(ENGINE), *args], capture_output=True).returncode == 2


def test_stored_values_agree_with_every_source():
    comp = rows("exact_complement.csv")
    for (n, m), v in comp.items():
        assert v >= complete_cost(m)
        assert all(v == r for r in references(n, m)), (n, m)
