"""Checks of the exact formulas for few columns and the exact values for few rows."""
from __future__ import annotations

import csv
import doctest
import os
from fractions import Fraction
from pathlib import Path

import pytest

import gradient_key.small_formulas as sf
from gradient_key.gradient_bound import complete_cost

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "small"
MS = range(4, 11)


def test_doctests():
    result = doctest.testmod(sf, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
    assert result.attempted > 0 and result.failed == 0


def test_fit_equals_enumeration_of_ordered_type_tuples():
    assert sf.load_formula(DATA, 4) == sf.direct_coefficients(4)


def test_numerator_is_an_integer_for_every_n():
    for m in MS:
        c = sf.load_formula(DATA, m)
        for n in range((m - 1).bit_length(), 40):
            v = sf.excess(c, m, n) * sf.falling(2**n, m)
            assert v.denominator == 1, f"m={m}, n={n}"


def test_formula_matches_brute_force_enumeration():
    # exact.csv comes from enumerating all matrices, not from the path formula the fit used
    cells = 0
    for r in csv.DictReader(open(ROOT / "data" / "gradient-bound" / "exact.csv")):
        n, m = int(r["n"]), int(r["m"])
        if 4 <= m <= 10:
            t = Fraction(int(r["numerator"]), int(r["denominator"]))
            assert t == complete_cost(m) + sf.excess(sf.load_formula(DATA, m), m, n), f"n={n}, m={m}"
            cells += 1
    assert cells >= 20, f"got [{cells}] cells"


def test_formula_matches_path_formula_cells_beyond_the_fit():
    cells = 0
    for r in csv.DictReader(open(ROOT / "data" / "gradient-bound" / "exact_formula.csv")):
        n, m = int(r["n"]), int(r["m"])
        if 4 <= m <= 10 and n > 2 ** (m - 1) // 2:
            assert Fraction(int(r["numerator"]), int(r["denominator"])) == complete_cost(m) + sf.excess(sf.load_formula(DATA, m), m, n)
            cells += 1
    assert cells > 0


@pytest.mark.parametrize("m", MS)
def test_leading_term_is_the_limit(m):
    raw = sf.load_formula(DATA, m)
    j, a, r = sf.leading_term(raw, m)
    n = 300
    assert abs(sf.excess(raw, m, n) / (a * r**n) - 1) < Fraction(1, 10**9)
    assert all(k <= j for k in raw)


def test_leading_terms_of_the_notes():
    got = {m: sf.leading_term(sf.load_formula(DATA, m), m)[1:] for m in (4, 5, 6)}
    assert got == {4: (Fraction(1, 4), Fraction(5, 8)), 5: (Fraction(2, 5), Fraction(3, 8)), 6: (Fraction(5, 18), Fraction(1, 2))}


def test_fixed_rows_match_the_formulas_and_vanish_at_the_complete_table():
    rows = sf.fixed_n_rows(DATA / "fixed_n.csv")
    for n, cells in rows.items():
        assert sorted(cells) == list(range(2, 2**n + 1)), f"n={n}"
        assert cells[2**n] == 0, f"n={n}"
        for m, v in cells.items():
            if 4 <= m <= 10:
                assert v == sf.excess(sf.load_formula(DATA, m), m, n), f"n={n}, m={m}"
    assert all(v == 0 for n in (1, 2) for v in rows[n].values())


def test_tables_in_the_article_are_current():
    art = Path(os.environ.get("ARTICLE_DIR", ROOT.parent / "gradient-key"))
    if not (art / "small_table_m.tex").exists():
        pytest.skip("the gradient-key repository is not checked out next to this one")
    assert (art / "small_table_m.tex").read_text() == sf.table_m_tex(DATA, range(4, 11))
    assert (art / "small_table_n.tex").read_text() == sf.table_n_tex(DATA / "fixed_n.csv")


def test_rate_of_the_subset_bound_is_sharp_only_for_m_4_and_8():
    from math import comb

    def beta(s):
        return Fraction(comb(s, s // 2) * (1 if s % 2 == 0 else 2), 2**s)

    same = [m for m in MS if sf.leading_term(sf.load_formula(DATA, m), m)[2] == max(1 - beta(s) for s in range(4, m + 1))]
    assert same == [4, 8], f"got [{same}]"
