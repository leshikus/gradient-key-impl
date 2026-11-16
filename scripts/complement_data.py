#!/usr/bin/env python3
"""Exact values near the complete cube by enumerating the missing words (engine build/cpp/gradient_complement).

    .venv/bin/python scripts/complement_data.py <n> <m> [<m> ...] [--threads T]

For each m, runs ``gradient_complement n 2^n-m T`` and appends the reduced value to
data/gradient-bound/exact_complement.csv (columns n,m,numerator,denominator,seconds). A pair already in the
file is skipped. The method and its proof are in the header of src/cpp/gradient_complement.cpp.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "build" / "cpp" / "gradient_complement"
OUT = ROOT / "data" / "gradient-bound" / "exact_complement.csv"


def run(n: int, m: int, threads: int) -> tuple[Fraction, float]:
    t0 = time.monotonic()
    line = subprocess.run([str(ENGINE), str(n), str(2**n - m), str(threads)], capture_output=True, text=True,
                          check=True).stdout.strip()
    nn, mm, num, den = line.split(",")
    if (int(nn), int(mm)) != (n, m):
        raise ValueError(f"engine answered {line!r} for (n,m)=({n},{m})")
    return Fraction(int(num), int(den)), time.monotonic() - t0


def main() -> None:
    args = sys.argv[1:]
    threads = len(os.sched_getaffinity(0))
    if "--threads" in args:
        i = args.index("--threads")
        threads = int(args[i + 1])
        del args[i : i + 2]
    n, ms = int(args[0]), [int(a) for a in args[1:]]
    done = set()
    if OUT.exists():
        done = {(int(r["n"]), int(r["m"])) for r in csv.DictReader(open(OUT))}
    else:
        OUT.write_text("n,m,numerator,denominator,seconds\n")
    for m in ms:
        if (n, m) in done:
            continue
        t, sec = run(n, m, threads)
        with open(OUT, "a") as f:
            f.write(f"{n},{m},{t.numerator},{t.denominator},{sec:.0f}\n")
        print(n, m, t, f"{sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
