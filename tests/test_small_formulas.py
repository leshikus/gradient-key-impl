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
MS = range(4, 14)


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
        if 4 <= m <= 13:
            t = Fraction(int(r["numerator"]), int(r["denominator"]))
            assert t == complete_cost(m) + sf.excess(sf.load_formula(DATA, m), m, n), f"n={n}, m={m}"
            cells += 1
    assert cells >= 20, f"got [{cells}] cells"


def test_formula_matches_path_formula_cells_beyond_the_fit():
    cells = 0
    for r in csv.DictReader(open(ROOT / "data" / "gradient-bound" / "exact_formula.csv")):
        n, m = int(r["n"]), int(r["m"])
        if 4 <= m <= 13 and n > 2 ** (m - 1) // 2:
            assert Fraction(int(r["numerator"]), int(r["denominator"])) == complete_cost(m) + sf.excess(sf.load_formula(DATA, m), m, n)
            cells += 1
    assert cells > 0


@pytest.mark.parametrize("m", MS)
def test_leading_term_is_the_limit(m):
    raw = sf.load_formula(DATA, m)
    j, a, r = sf.leading_term(raw, m)
    n = max(300, 60 * j)  # the next term falls like ((J-1)/J)^n
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
            if 4 <= m <= 13:
                assert v == sf.excess(sf.load_formula(DATA, m), m, n), f"n={n}, m={m}"
    assert all(v == 0 for n in (1, 2) for v in rows[n].values())


def test_tables_in_the_article_are_current():
    art = Path(os.environ.get("ARTICLE_DIR", ROOT.parent / "gradient-key"))
    if not (art / "small_table_m.tex").exists():
        pytest.skip("the gradient-key repository is not checked out next to this one")
    assert (art / "small_table_m.tex").read_text() == sf.table_m_tex(DATA, range(4, 14))
    assert (art / "small_table_n.tex").read_text() == sf.table_n_tex(DATA / "fixed_n.csv")


def test_rate_of_the_subset_bound_is_sharp_only_for_m_4_and_8():
    from math import comb

    def beta(s):
        return Fraction(comb(s, s // 2) * (1 if s % 2 == 0 else 2), 2**s)

    same = [m for m in MS if sf.leading_term(sf.load_formula(DATA, m), m)[2] == max(1 - beta(s) for s in range(4, m + 1))]
    assert same == [4, 8], f"got [{same}]"


ENGINE = ROOT / "build" / "cpp" / "gradient_bound"


def reference_sums() -> dict[int, dict[int, int]]:
    """`engine_reference.csv`: totals of the original `formula` engine, recorded as a regression reference."""
    out: dict[int, dict[int, int]] = {}
    for r in csv.DictReader(open(DATA / "engine_reference.csv")):
        out.setdefault(int(r["m"]), {})[int(r["n"])] = int(r["depth_sum"])
    return out


def test_reference_matches_the_formulas():
    for m, cells in reference_sums().items():
        if 4 <= m <= 13:
            c = sf.load_formula(DATA, m)
            for n, v in cells.items():
                assert sf.excess_numerator_from_sum(n, m, v) == sf.numerator(c, n), f"n={n}, m={m}"


@pytest.mark.skipif(not ENGINE.exists(), reason="build/cpp/gradient_bound is not built")
@pytest.mark.parametrize("m", range(2, 13))
def test_formula_all_matches_the_reference(m):
    cells = reference_sums()[m]
    got = sf.engine_depth_sums(max(cells), m, ENGINE)
    assert {n: got[n] for n in cells} == cells
    assert min(got) == (m - 1).bit_length() or m == 2


@pytest.mark.skipif(not ENGINE.exists(), reason="build/cpp/gradient_bound is not built")
@pytest.mark.parametrize("n,m", [(5, 7), (12, 10), (40, 12)])
def test_formula_matches_the_reference(n, m):
    assert sf.engine_excess_numerator(n, m, ENGINE) == sf.excess_numerator_from_sum(n, m, reference_sums()[m][n])


def test_markdown_tables_are_current():
    assert (DATA / "formulas.md").read_text() == sf.formulas_md(DATA, range(2, 14))
    gb = ROOT / "data" / "gradient-bound"
    assert (DATA / "exact_values.md").read_text() == sf.exact_values_md(gb / "exact_formula.csv", 12, 13,
                                                                        complement=gb / "exact_complement.csv")


def test_count_of_exact_values_in_the_article():
    tex = Path(os.environ.get("ARTICLE_DIR", ROOT.parent / "gradient-key")) / "constants.tex"
    if not tex.exists():
        pytest.skip("the gradient-key repository is not checked out next to this one")
    n_big = sum(1 for r in csv.DictReader(open(ROOT / "data" / "gradient-bound" / "exact_formula.csv")) if int(r["n"]) <= 12 and int(r["m"]) > 12)
    assert rf"\newcommand{{\exactcellsbig}}{{{n_big}}}" in tex.read_text()


@pytest.mark.skipif(not ENGINE.exists(), reason="build/cpp/gradient_bound is not built")
@pytest.mark.parametrize("m", range(4, 13))
def test_engine_matches_the_exact_formulas(m):
    """The path-formula engine against the closed formulas for m <= 12, at n beyond the stored reference."""
    c = sf.load_formula(DATA, m)
    for n in sorted({(m - 1).bit_length(), 2 * m, 3 * m + 1, 50}):
        assert sf.engine_excess_numerator(n, m, ENGINE) == sf.numerator(c, n), f"n={n}, m={m}"


def test_closed_forms_near_the_complete_cube():
    """Propositions dense and near-dense against every stored exact value they cover, and the control values."""
    controls = {(3, 4): Fraction(1, 35), (4, 11): Fraction(1, 3003), (5, 26): Fraction(1, 736281),
                (4, 10): Fraction(18, 5005), (5, 25): Fraction(2, 90675)}
    for (n, m), d in controls.items():
        assert sf.closed_form_excess(n, m)[0] == d, (n, m)
    exact = {}
    for name in ("exact.csv", "exact_formula.csv"):
        for r in csv.DictReader(open(ROOT / "data" / "gradient-bound" / name)):
            exact[int(r["n"]), int(r["m"])] = Fraction(int(r["numerator"]), int(r["denominator"])) - complete_cost(int(r["m"]))
    for n, cells in sf.fixed_n_rows(DATA / "fixed_n.csv").items():
        for m, v in cells.items():
            exact[n, m] = v
    checked = 0
    for (n, m), v in exact.items():
        cf = sf.closed_form_excess(n, m)
        if cf is not None:
            assert cf[0] == v, (n, m)
            checked += 1
    assert checked >= 24, checked
    # the cases r = n+1 and r = n+2 are positive: Proposition dense is sharp
    assert all(sf.closed_form_excess(n, 2**n - n - k)[0] > 0 for n in range(4, 13) for k in (1, 2))
