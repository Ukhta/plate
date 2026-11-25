"""Геометрия поверочной плиты и измерительной сетки.

Сетка ``(N+1) x (M+1)`` узлов:

* ось X — длинная сторона, индексы-цифры ``0..N``, отсчёт от 0;
* ось Y — короткая сторона, индексы-буквы ``A..``, отсчёт от A.

Координаты узлов::

    X_i = indent + i * (length - 2*indent) / N
    Y_j = indent + j * (width  - 2*indent) / M

Точка ``A0`` — левый нижний угол ``(indent, indent)``,
``A6`` (по умолчанию) — правый нижний ``(length - indent, indent)``.
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple


@lru_cache(maxsize=1)
def _alphabet() -> Tuple[str, ...]:
    letters = list(string.ascii_uppercase)
    letters += [f"{a}{b}" for a in string.ascii_uppercase for b in string.ascii_uppercase]
    return tuple(letters)


def node_name(i: int, j: int) -> str:
    """Имя узла: буква (строка Y, индекс ``j``) + цифра (столбец X, индекс ``i``).

    Примеры: ``node_name(0, 0) == 'A0'``, ``node_name(6, 0) == 'A6'``.
    """
    if i < 0 or j < 0:
        raise ValueError(f"Отрицательный индекс узла: i={i}, j={j}")
    letters = _alphabet()
    if j >= len(letters):
        raise ValueError(f"Индекс строки {j} вне поддерживаемого диапазона имён")
    return f"{letters[j]}{i}"


def parse_node_name(name: str) -> Tuple[int, int]:
    """Разобрать имя узла вида ``A0``/``AB12`` в пару индексов ``(i, j)``."""
    s = str(name).strip().upper()
    k = 0
    while k < len(s) and s[k].isalpha():
        k += 1
    if k == 0 or k == len(s):
        raise ValueError(f"Некорректное имя точки: {name!r} (ожидалось вида A0)")
    letter, digits = s[:k], s[k:]
    try:
        j = _alphabet().index(letter)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Неизвестное имя строки: {letter!r} в точке {name!r}") from exc
    try:
        i = int(digits)
    except ValueError as exc:
        raise ValueError(f"Некорректный номер столбца в точке {name!r}") from exc
    return i, j


@dataclass(frozen=True)
class GridGeometry:
    """Геометрия плиты и равномерной измерительной сетки."""

    length_mm: float = 1000.0
    width_mm: float = 630.0
    indent_mm: float = 20.0
    n_intervals: int = 6  # интервалов по длинной стороне (X)
    m_intervals: int = 6  # интервалов по короткой стороне (Y)

    def __post_init__(self) -> None:
        object.__setattr__(self, "length_mm", float(self.length_mm))
        object.__setattr__(self, "width_mm", float(self.width_mm))
        object.__setattr__(self, "indent_mm", float(self.indent_mm))
        if self.length_mm <= 0 or self.width_mm <= 0:
            raise ValueError("Габариты плиты должны быть положительными")
        if self.n_intervals < 1 or self.m_intervals < 1:
            raise ValueError("Число интервалов сетки должно быть >= 1")
        if not (0 <= self.indent_mm < min(self.length_mm, self.width_mm) / 2):
            raise ValueError("Отступ от края должен лежать в диапазоне [0, min(L,W)/2)")

    # ------------------------------------------------------------------ size
    @property
    def nx(self) -> int:
        """Число узлов по оси X."""
        return self.n_intervals + 1

    @property
    def ny(self) -> int:
        """Число узлов по оси Y."""
        return self.m_intervals + 1

    @property
    def num_nodes(self) -> int:
        return self.nx * self.ny

    @property
    def step_x_mm(self) -> float:
        """Шаг сетки по длинной стороне, мм."""
        return (self.length_mm - 2 * self.indent_mm) / self.n_intervals

    @property
    def step_y_mm(self) -> float:
        """Шаг сетки по короткой стороне, мм."""
        return (self.width_mm - 2 * self.indent_mm) / self.m_intervals

    # ----------------------------------------------------------- coordinates
    def x(self, i: int) -> float:
        return self.indent_mm + i * self.step_x_mm

    def y(self, j: int) -> float:
        return self.indent_mm + j * self.step_y_mm

    def xs(self) -> List[float]:
        return [self.x(i) for i in range(self.nx)]

    def ys(self) -> List[float]:
        return [self.y(j) for j in range(self.ny)]

    def node_xy(self, i: int, j: int) -> Tuple[float, float]:
        return self.x(i), self.y(j)

    # ------------------------------------------------------------------ names
    def node_index(self, name: str) -> Tuple[int, int]:
        """Индексы ``(i, j)`` узла по имени с проверкой попадания в сетку."""
        i, j = parse_node_name(name)
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            raise ValueError(
                f"Точка {name!r} вне сетки {self.nx}x{self.ny} "
                f"(X: 0..{self.nx - 1}, Y: A..{node_name(0, self.ny - 1)[0]})"
            )
        return i, j

    def flat_index(self, i: int, j: int) -> int:
        """Линейный индекс узла (строки Y снизу вверх)."""
        return j * self.nx + i

    def name_matrix(self) -> List[List[str]]:
        """Имена узлов таблицей ``[строка сверху вниз][столбец]`` (для печати)."""
        return [[node_name(i, self.ny - 1 - r) for i in range(self.nx)]
                for r in range(self.ny)]

    def nodes_by_name(self) -> Dict[str, Tuple[int, int]]:
        return {node_name(i, j): (i, j)
                for i in range(self.nx) for j in range(self.ny)}

    # -------------------------------------------------------------- profiles
    def grid_steps(self, start: str, end: str) -> int:
        """Число шагов сетки между двумя узлами (для диагонали — max(|di|,|dj|))."""
        i0, j0 = self.node_index(start)
        i1, j1 = self.node_index(end)
        return max(abs(i1 - i0), abs(j1 - j0))

    def expected_points(self, start: str, end: str) -> int:
        """Расчётное число угловых отсчётов профиля ``start``→``end``.

        Профиль из ``K`` отсчётов задаёт ``K+1`` равных участка по хорде.
        Для линий сетки (горизонталей и вертикалей) внутренние отсчёты
        попадают в промежуточные узлы, поэтому ``K = steps - 1``.  Для
        диагоналей внутренние узлы не совпадают с позициями уровня
        (ступенчатая база ``step_x`` ≠ ``step_y``), поэтому используется
        полное число шагов ``K = max(|di|, |dj|)`` — так же, как принято в
        методике измерений плоскостности по диагоналям.
        """
        i0, j0 = self.node_index(start)
        i1, j1 = self.node_index(end)
        di, dj = abs(i1 - i0), abs(j1 - j0)
        steps = self.grid_steps(start, end)
        if di == 0 or dj == 0:          # линия сетки
            return steps - 1
        return steps                    # диагональ

    def chord_length_mm(self, start: str, end: str) -> float:
        """Расстояние между узлами, мм."""
        x0, y0 = self.node_xy(*self.node_index(start))
        x1, y1 = self.node_xy(*self.node_index(end))
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

    def is_valid_profile(self, start: str, end: str) -> bool:
        """Профиль допустим, если концы — разные узлы одной линии сетки или диагонали."""
        try:
            i0, j0 = self.node_index(start)
            i1, j1 = self.node_index(end)
        except ValueError:
            return False
        if (i0, j0) == (i1, j1):
            return False
        di, dj = abs(i1 - i0), abs(j1 - j0)
        return di == 0 or dj == 0 or di == dj

    def along_coords(self, start: str, end: str, k: int) -> List[Tuple[float, float]]:
        """Координаты ``k`` внутренних точек профиля (равномерно между узлами)."""
        x0, y0 = self.node_xy(*self.node_index(start))
        x1, y1 = self.node_xy(*self.node_index(end))
        return [(x0 + (x1 - x0) * t / (k + 1), y0 + (y1 - y0) * t / (k + 1))
                for t in range(1, k + 1)]
