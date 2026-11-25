"""CLI: ``python -m levelness input.txt --config config.json --out ./report``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import Config, VALID_CLASSES
from .parser import ParseError, parse_file
from .report import write_all
from .solver import process

logger = logging.getLogger("levelness")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m levelness",
        description="Расчёт плоскостности поверочных плит по данным "
                    "электронного уровня (ISO 8512 / DIN 876).")
    p.add_argument("input", type=Path,
                   help="текстовый файл профилей (см. example_input.txt)")
    p.add_argument("--config", type=Path, default=None,
                   help="JSON-файл конфигурации (габариты, сетка, класс)")
    p.add_argument("--out", type=Path, default=Path("report"),
                   help="каталог результатов (по умолчанию ./report)")
    p.add_argument("--length", dest="length_mm", type=float,
                   help="длина плиты, мм (переопределяет конфиг)")
    p.add_argument("--width", dest="width_mm", type=float,
                   help="ширина плиты, мм")
    p.add_argument("--indent", dest="indent_mm", type=float,
                   help="отступ первой/последней точки от края, мм")
    p.add_argument("--n-intervals", dest="n_intervals", type=int,
                   help="интервалов сетки по длинной стороне")
    p.add_argument("--m-intervals", dest="m_intervals", type=int,
                   help="интервалов сетки по короткой стороне")
    p.add_argument("--class", dest="precision_class", choices=VALID_CLASSES,
                   help="класс точности плиты (00, 0, 1, 2)")
    p.add_argument("--material", choices=("cast_iron", "granite"),
                   help="материал плиты")
    p.add_argument("--point-tolerance", type=int, default=1,
                   help="допуск расхождения числа точек профиля с расчётным "
                        "(±N, по умолчанию 1)")
    p.add_argument("--all-diagonals", action="store_true",
                   help="рассчитать прямолинейность всех коротких диагоналей")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="подробный вывод (DEBUG)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s")

    try:
        cfg = Config.load(args.config)
        overrides = {k: getattr(args, k) for k in
                     ("length_mm", "width_mm", "indent_mm", "n_intervals",
                      "m_intervals", "precision_class", "material")}
        cfg = cfg.with_overrides({k: v for k, v in overrides.items()
                                  if v is not None})
        profiles = parse_file(args.input, cfg.geometry,
                              point_tolerance=args.point_tolerance)
        result = process(profiles, cfg.geometry,
                         include_all_diagonals=args.all_diagonals)
        write_all(result, cfg, profiles, args.out)
    except (ParseError, FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except Exception:  # noqa: BLE001 — показываем стек только в verbose
        logger.exception("Необработанная ошибка расчёта")
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
