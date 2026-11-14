# gradient-key-impl

Code, tests and data of the article *The gradient key on random tables* (private repository `leshikus/gradient-key`, checked out next to this one). MIT licence.
The code was moved here from `leshikus/keytrees`; the history is there.

- `src/cpp/`: engines in C++23 (`gradient_bound`: exact path formula; `gradient_enum`: enumeration; `gradient_tplus`: interval checks).
- `src/python/gradient_key/`: everything else, with doctests.
- `scripts/`: regenerate the data (`gradient_bound_data.py`, `gradient_analytic_9_data.py`) and re-check the certificates (`verify_certificates.py`).
- `tests/`: checks of the numbers in the article. `data/gradient-bound/`: the data behind every table.

```sh
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
cmake -S src/cpp -B build/cpp -G Ninja -DCMAKE_CXX_COMPILER=clang++ && cmake --build build/cpp
.venv/bin/python -m pytest -q          # slow checks: -m ""
```

Generated TeX goes to `$ARTICLE_DIR` (default `../gradient-key`). `requirements-certificates.txt` pins the versions of the interval computations.
