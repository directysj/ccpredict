"""Command line.

The notebook's two-stage pipeline runner lived here; this branch has only the
one-stage regressor, so there is nothing for it to run. The two entry points
that remain are:

    python -m ccpredict.benchmark --write-config bench.json
    python -m ccpredict.benchmark --config bench.json
    python -m ccpredict.simulate OUT_DIR

This module forwards to the benchmark so `python -m ccpredict` and the
`ccpredict` console script keep working.
"""
from __future__ import annotations

from .benchmark import main

if __name__ == "__main__":
    main()
