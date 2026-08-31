"""CLI de MapForge.

Uso::

    python -m mapforge generate -c config/valle_bonito.yaml [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from mapforge import __version__
from mapforge.generator import Generator
from mapforge.project import Project, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapforge",
        description="MapForge: generador local de mapas FS25 (heightmap + OSM + template).",
    )
    parser.add_argument(
        "--version", action="version", version=f"mapforge {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="Genera un mapa a partir de un config.yaml"
    )
    generate.add_argument(
        "-c",
        "--config",
        required=True,
        metavar="CONFIG_YAML",
        help="Ruta al config.yaml del proyecto",
    )
    generate.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Logging DEBUG en lugar de INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        setup_logging(logging.DEBUG if args.verbose else logging.INFO)
        project = Project.from_yaml(args.config)
        Generator(project).run()
        return 0

    parser.error(f"comando desconocido: {args.command}")
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())
