"""Regression tests for the source review of 2026-11-15 (GK-01 ... GK-12)."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import mpmath
import pytest
from mpmath import iv, libmp

import gradient_key.gradient_bound as gb

ROOT = Path(__file__).resolve().parents[1]
BUILD = Path(os.environ.get("GRADIENT_KEY_BUILD", ROOT / "build" / "cpp"))


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _engine(name, *args):
    path = BUILD / name
    if not path.exists():
        pytest.skip(f"build/cpp/{name} is not built")
    return subprocess.run([str(path), *map(str, args)], capture_output=True, text=True)


def test_gk01_upper_endpoint_is_not_rounded_down():
    iv.prec = 64
    x = iv.mpf(1) / 3
    rounded = mpmath.mpf(x.b)  # the old conversion: nearest at mp.prec
    assert Fraction(*libmp.to_rational(rounded._mpf_)) < Fraction(1, 3)
    assert Fraction(*libmp.to_rational(gb._upper(x)._mpf_)) >= Fraction(1, 3)


def test_gk02_decimal_ceiling_is_exact():
    up = _script("gradient_bound_data").up
    assert up(Fraction(10000000000005, 10**13), 3) == "1.001"
    assert up(Fraction(1), 3) == "1.000"
    assert up(Fraction(1000001, 10**6), 3) == "1.001"
    assert up(Fraction(-1, 3), 2) == "-0.33"
    with pytest.raises(TypeError):
        up(1.0, 3)


GOOD = "3,4,4,14,4611686018427387847:3408,4611686018427387817:3408"


@pytest.mark.parametrize("line", [
    "3,4,4,14,4611686018427387847:3408,4611686018427387817:3409",  # spare residue corrupted
    "3,4,4,14,4611686018427387847:3408",                             # no spare prime
    "3,4,4,14,4611686018427387847:3408,4611686018427387847:3408",    # moduli not coprime
    "3,4,4,14,4611686018427387847:x,4611686018427387817:3408",       # malformed
    "3,4,4,14,7:3,4611686018427387817:3",                            # modulus too small for the range
])
def test_gk07_crt_rejects_bad_lines(line):
    assert gb.crt_combine(GOOD) == (3, 4, 3408)
    with pytest.raises(ValueError):
        gb.crt_combine(line)


def test_gk07_crt_checks_survive_python_O():
    code = ("from gradient_key.gradient_bound import crt_combine\n"
            "try:\n crt_combine('3,4,4,14,4611686018427387847:3408,4611686018427387817:3409')\n"
            "except ValueError:\n raise SystemExit(0)\nraise SystemExit(1)\n")
    env = dict(os.environ, PYTHONOPTIMIZE="1", PYTHONPATH=str(ROOT / "src" / "python"))
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0


def test_gk04_gk05_engines_reject_overflowing_cells():
    assert _engine("gradient_enum", "sym", 1, 36, 1).returncode == 2
    assert _engine("gradient_enum", "sym", 7, 18, 1).returncode == 2  # n m + bit_length(m) = 131 > 126
    assert _engine("gradient_bound", "exact", 20, 20).returncode == 2
    assert _engine("gradient_bound", "exact", "3", "4x").returncode == 2
    assert _engine("gradient_enum", "sym", "4", "8x", "1").returncode == 2


def test_gk09_engine_fails_on_missing_data(tmp_path):
    out = _engine("gradient_tplus", "cell", 30, 2**20, "--data", tmp_path)
    assert out.returncode == 3 and "data error" in out.stderr


@pytest.mark.parametrize("cell", [(3, 4), (3, 5), (4, 6), (4, 9)])
def test_cross_engine_exact_values(cell):
    """Python brute force, Python path formula, C++ enumeration and the C++ modular formula agree exactly."""
    n, m = cell
    brute = gb.exact_average(n, m)
    assert gb.path_formula(n, m) == brute
    ex = _engine("gradient_bound", "exact", n, m).stdout.strip().split(",")
    assert Fraction(int(ex[2]), int(ex[3])) == brute
    _, _, value = gb.crt_combine(_engine("gradient_bound", "formula", n, m, 1).stdout.strip().splitlines()[-1])
    tables = 1
    for i in range(m):
        tables *= 2**n - i
    assert Fraction(value, tables) == brute


def test_formula_is_deterministic_across_threads():
    outs = {t: _engine("gradient_bound", "formula", 5, 12, t).stdout for t in (1, 2, 4)}
    assert outs[1] == outs[2] == outs[4]


def test_gk11_no_global_precision_side_effects():
    dps = mpmath.mp.dps
    code = "import mpmath; d = mpmath.mp.dps; import gradient_key.gradient_analytic_9; raise SystemExit(mpmath.mp.dps != d)"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src" / "python"))
    assert subprocess.run([sys.executable, "-c", code], env=env).returncode == 0
    iv.prec = 53
    import gradient_key.gradient_tplus as gt

    gt.uniform_bound(30, 2**20)
    gb.cylinder_tail(500, 1)
    assert iv.prec == 53 and mpmath.mp.dps == dps


def test_gk12_domain_errors():
    with pytest.raises(ValueError):
        gb.gradient_cost([], 3)
    with pytest.raises(ValueError):
        gb.exact_average(2, 5)
    with pytest.raises(ValueError):
        gb.path_formula(0, 1)
