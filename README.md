# gradient-key-impl

Code, tests and data of the article *The gradient key on random tables* (private repository `leshikus/gradient-key`, checked out next to this one). MIT licence.
The code was moved here from `leshikus/keytrees`; the history is there.

- `src/cpp/`: engines in C++23 (`gradient_bound`: exact path formula, `formula-all` gives every `n <= nmax` in one parallel pass; `gradient_enum`: enumeration; `gradient_tplus`: interval checks).
- `src/python/gradient_key/`: everything else, with doctests.
- `scripts/`: regenerate the data (`gradient_bound_data.py`, `gradient_analytic_9_data.py`) and re-check the certificates (`verify_certificates.py`).
- `tests/`: checks of the numbers in the article. `data/gradient-bound/`: the data behind every table.

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cmake -S src/cpp -B build/cpp -G Ninja -DCMAKE_CXX_COMPILER=clang++ && cmake --build build/cpp
.venv/bin/python -m pytest -q          # slow checks: -m ""
```

Generated TeX goes to `$ARTICLE_DIR` (default `../gradient-key`). `requirements-certificates.txt` pins the versions of the interval computations.

## What each result of the article rests on

The article (commit `a6b9060` of `leshikus/gradient-key` and later) has three results. Each is either a proof, an exact
finite computation, or both; nothing is estimated by sampling.

| Result | Kind of evidence | Code and data | Checks |
|---|---|---|---|
| (1) Exact formulas for `m <= 12` (Proposition 3.2) | proof of the form for every `m`; coefficients `a_j` by exact integer computation | `small_formulas.py`, `data/small/formula_m*.json`, `formulas.md` | fit checked on 6 to 8 extra `n` per `m`; `m = 4` also directly; `tests/test_small_formulas.py` |
| (2) Exact values, `m > 12`, `n <= 12` (Proposition 3.1) | exact finite computation (integers modulo 62-bit primes, Chinese remainders with a spare prime as a check, range check `N <= T <= (m-1)N`) | `gradient_bound formula`, `gradient_bound.crt_combine`, `data/gradient-bound/exact_formula.csv`, `data/small/exact_values.md` | brute force on all 48 cells of `exact.csv` (`n <= 7`); Python path formula on small cells |
| (3) Theorem 5.5, `t < log m + 2 + 2 ln(2l)`, `t < log m + 2` for `n >= 2l - 2` | proof; no computation | none | sanity check of the per-table inequality: `scripts/check_uniform_recursion.py` |

The constants `c_K`, `c_b` and the table `t^+` (`gradient_constants.csv`, `blocked_constants.csv`, `tplus.csv`) belong to
earlier versions of the article. They are upper bounds computed in `mpmath` interval arithmetic (`verify_certificates.py`,
`certify_tplus.py`); the C++ engine `gradient_tplus` uses directed rounding and widens the C library's `log`, `log1p`,
`exp` by 16 ulp, which is checked at run time on a few values but not proved for every C library. A finite grid of
values, exact or bounded, is never evidence for a statement about all `n` and `m`.
The modules `gradient_analytic_*.py` and their tests check the working notes `analytic/*.tex` of those earlier versions;
the notes are in the history of `leshikus/gradient-key` (commit `8ef012e` and before).

Toolchain used for the data: x86_64, GCC 15.2.0 / Clang, glibc 2.43, CMake >= 3.28, Python 3.14.4 with the pins of
`requirements-certificates.txt`. `scripts/verify_certificates.py --strict` fails on any skipped check or version mismatch.
