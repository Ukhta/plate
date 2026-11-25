"""Конфигурация расчёта: JSON-файл + переопределение аргументами CLI.

Пример ``config.json``::

    {
      "plate": {"length_mm": 1000, "width_mm": 630, "indent_mm": 20},
      "grid": {"n_intervals": 6, "m_intervals": 6},
      "precision_class": "1",
      "material": "cast_iron"
    }
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Optional

from .grid import GridGeometry

logger = logging.getLogger(__name__)

#: Допуски плоскостности, мкм, по классам точности (упрощённые таблицы
#: DIN 876 / ISO 8512 для типоразмеров плит; ключ — диагональ/диаметр опис.
#: окружности в мм).  Для нестандартных размеров выполняется линейная
#: интерполяция по ближайшим строкам таблицы класса.
TOLERANCES_UM: Dict[str, Dict[int, int]] = {
    # class -> {nominal size (mm, approx. diagonal): flatness tolerance um}
    "00": {630: 2, 1000: 3, 1250: 3, 1600: 4, 2000: 4, 2500: 5, 3150: 6, 4000: 7},
    "0": {630: 4, 1000: 6, 1250: 6, 1600: 8, 2000: 9, 2500: 11, 3150: 13, 4000: 16},
    "1": {630: 8, 1000: 11, 1250: 12, 1600: 16, 2000: 18, 2500: 22, 3150: 26, 4000: 32},
    "2": {630: 15, 1000: 22, 1250: 25, 1600: 32, 2000: 36, 2500: 45, 3150: 54, 4000: 67},
}

VALID_CLASSES = ("00", "0", "1", "2")


@dataclass
class Config:
    """Параметры расчёта плоскостности."""

    geometry: GridGeometry = field(default_factory=GridGeometry)
    precision_class: str = "1"
    material: str = "cast_iron"          # cast_iron | granite
    #: множитель допуска для гранитных плит (гранитные плиты обычно строже)
    granite_tolerance_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.precision_class not in VALID_CLASSES:
            raise ValueError(
                f"Класс точности {self.precision_class!r} недопустим, "
                f"ожидается один из {VALID_CLASSES}"
            )
        if self.material not in ("cast_iron", "granite"):
            raise ValueError("material должен быть 'cast_iron' или 'granite'")

    # ------------------------------------------------------------ tolerance
    def tolerance_um(self) -> int:
        """Допуск плоскостности для данного размера плиты и класса точности."""
        table = TOLERANCES_UM[self.precision_class]
        sizes = sorted(table)
        # характерный размер — диагональ сетки
        diag = ((self.geometry.length_mm - 2 * self.geometry.indent_mm) ** 2
                + (self.geometry.width_mm - 2 * self.geometry.indent_mm) ** 2) ** 0.5
        if diag <= sizes[0]:
            tol = table[sizes[0]]
        elif diag >= sizes[-1]:
            tol = table[sizes[-1]]
        else:  # лог-интерполяция между соседними типоразмерами
            import math
            lo = max(s for s in sizes if s <= diag)
            hi = min(s for s in sizes if s >= diag)
            if lo == hi:
                tol = table[lo]
            else:
                t = (math.log(diag) - math.log(lo)) / (math.log(hi) - math.log(lo))
                tol = round(table[lo] + t * (table[hi] - table[lo]))
        return max(1, round(tol * self.granite_tolerance_factor))

    # -------------------------------------------------------------- loading
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        plate = data.get("plate", {})
        grid = data.get("grid", {})
        geometry = GridGeometry(
            length_mm=float(plate.get("length_mm", 1000.0)),
            width_mm=float(plate.get("width_mm", 630.0)),
            indent_mm=float(plate.get("indent_mm", 20.0)),
            n_intervals=int(grid.get("n_intervals", data.get("n_intervals", 6))),
            m_intervals=int(grid.get("m_intervals", data.get("m_intervals", 6))),
        )
        return cls(
            geometry=geometry,
            precision_class=str(data.get("precision_class", "1")),
            material=str(data.get("material", "cast_iron")),
            granite_tolerance_factor=float(data.get("granite_tolerance_factor", 1.0)),
        )

    @classmethod
    def load(cls, path: Optional[Path]) -> "Config":
        """Загрузить конфигурацию из JSON-файла (или значения по умолчанию)."""
        if path is None:
            logger.debug("Файл конфигурации не задан — используются значения по умолчанию")
            return cls()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {p}")
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ошибка разбора JSON-конфигурации {p}: {exc}") from exc
        cfg = cls.from_dict(data)
        logger.info("Конфигурация загружена из %s", p)
        return cfg

    def with_overrides(self, args: Dict[str, Any]) -> "Config":
        """Применить непустые переопределения из аргументов CLI."""
        geom_fields = {k: v for k, v in args.items()
                       if k in ("length_mm", "width_mm", "indent_mm",
                                "n_intervals", "m_intervals") and v is not None}
        geometry = replace(self.geometry, **geom_fields) if geom_fields else self.geometry
        overrides: Dict[str, Any] = {"geometry": geometry}
        for key in ("precision_class", "material"):
            if args.get(key) is not None:
                overrides[key] = args[key]
        return Config(**{**{"geometry": self.geometry,
                            "precision_class": self.precision_class,
                            "material": self.material,
                            "granite_tolerance_factor": self.granite_tolerance_factor},
                         **overrides})

    def to_dict(self) -> Dict[str, Any]:
        g = self.geometry
        return {
            "plate": {"length_mm": g.length_mm, "width_mm": g.width_mm,
                      "indent_mm": g.indent_mm},
            "grid": {"n_intervals": g.n_intervals, "m_intervals": g.m_intervals,
                     "nodes": f"{g.nx}x{g.ny}",
                     "step_x_mm": g.step_x_mm, "step_y_mm": g.step_y_mm},
            "precision_class": self.precision_class,
            "material": self.material,
            "tolerance_um": self.tolerance_um(),
        }
