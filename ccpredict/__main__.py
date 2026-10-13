"""Command line: python -m ccpredict --config config.json

python -m ccpredict --write-config config.json   writes the default (notebook) config to edit.
"""
from __future__ import annotations

import argparse
import logging

from .config import PipelineConfig
from .pipeline import CellCyclePipeline


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="ccpredict", description=__doc__)
    parser.add_argument("--config", help="JSON config; defaults reproduce the notebook")
    parser.add_argument("--write-config", metavar="PATH", help="write the default config and exit")
    parser.add_argument("--output-dir", help="override paths.output_dir")
    parser.add_argument("--no-regressor", action="store_true", help="stop after stage 1")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    if args.write_config:
        PipelineConfig().to_json(args.write_config)
        print(f"wrote {args.write_config}")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import matplotlib

    matplotlib.use("Agg")
    cfg = PipelineConfig.from_json(args.config) if args.config else PipelineConfig()
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
    CellCyclePipeline(cfg).run(with_regressor=not args.no_regressor,
                               make_figures=not args.no_figures)


if __name__ == "__main__":
    main()
