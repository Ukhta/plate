"""Общие фикстуры pytest: путь к пакету и эталонная конфигурация."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from levelness.config import Config          # noqa: E402
from levelness.grid import GridGeometry      # noqa: E402


@pytest.fixture()
def geometry() -> GridGeometry:
    """Стандартная геометрия из ТЗ: 1000×630 мм, отступ 20, сетка 6×6."""
    return GridGeometry(length_mm=1000, width_mm=630, indent_mm=20,
                        n_intervals=6, m_intervals=6)


@pytest.fixture()
def config(geometry: GridGeometry) -> Config:
    return Config(geometry=geometry, precision_class="1", material="cast_iron")


@pytest.fixture()
def example_file() -> Path:
    return ROOT / "example_input.txt"
