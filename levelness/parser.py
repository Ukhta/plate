"""Разбор текстового файла профилей электронного уровня.

Формат строки (один профиль на строку)::

    <начальная точка> <конечная точка> <время измерения> <шаг измерения> <массив углов>

Пример::

    A0 A6 2026-11-18T14:32 160 0.5 0.4 0.3 0.35 0.42 0.5

Допускаются разделители пробел/табуляция; массив углов может быть задан
через пробел или через запятую.  Пустые строки и строки, начинающиеся с
``#`` или ``;``, игнорируются.  Ошибки сопровождаются номером строки.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Union

from .grid import GridGeometry

logger = logging.getLogger(__name__)


class ParseError(ValueError):
    """Ошибка формата входного файла (с указанием номера строки)."""

    def __init__(self, line_no: int, message: str) -> None:
        super().__init__(f"Строка {line_no}: {message}")
        self.line_no = line_no
        self.raw_message = message


@dataclass
class Profile:
    """Один измеренный профиль между двумя узлами сетки."""

    start: str                       # имя узла, напр. 'A0'
    end: str                         # имя узла, напр. 'A6'
    timestamp: datetime
    step_mm: float                   # база (шаг) измерения уровня, мм
    angles_um_per_m: List[float] = field(default_factory=list)  # θ, мкм/м

    @property
    def num_points(self) -> int:
        return len(self.angles_um_per_m)

    @property
    def chord_mm(self) -> float:
        """Фактическая длина базы профиля (профиль измеряется «встык»)."""
        return self.step_mm * (self.num_points + 1)

    def height_increments_um(self) -> List[float]:
        """Перепады высот за шаг: Δh_k = step_mm · θ_k · 1e-6 [м] → мкм."""
        return [self.step_mm * theta for theta in self.angles_um_per_m]

    def cumulative_heights_um(self) -> List[float]:
        """Накопленные высоты относительно начала профиля, мкм (K+1 значений)."""
        h, out = 0.0, [0.0]
        for dh in self.height_increments_um():
            h += dh
            out.append(h)
        return out


def _parse_timestamp(text: str, line_no: int) -> datetime:
    for fmt in ("%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ParseError(line_no, f"не удалось разобрать время {text!r} "
                              "(ожидалось, напр., 2026-11-18T14:32)")


def _parse_angles(tokens: List[str], line_no: int) -> List[float]:
    values: List[float] = []
    for tok in tokens:
        for part in tok.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                values.append(float(part.replace(",", ".", 1) if part.count(",") == 1 else part))
            except ValueError as exc:
                raise ParseError(line_no,
                                 f"некорректное значение угла {part!r} в массиве") from exc
    if not values:
        raise ParseError(line_no, "пустой массив углов")
    return values


def parse_line(line: str, line_no: int) -> Profile:
    """Разобрать одну строку файла в :class:`Profile`."""
    tokens = line.split()
    if len(tokens) < 5:
        raise ParseError(line_no,
                         f"ожидались поля: <старт> <финиш> <время> <шаг> <углы...>, "
                         f"получено {len(tokens)} токенов")
    start, end, time_s, step_s = tokens[:4]
    try:
        step_mm = float(step_s)
    except ValueError as exc:
        raise ParseError(line_no, f"некорректный шаг измерения {step_s!r}") from exc
    if step_mm <= 0:
        raise ParseError(line_no, f"шаг измерения должен быть > 0, получено {step_mm}")
    profile = Profile(
        start=start.upper(),
        end=end.upper(),
        timestamp=_parse_timestamp(time_s, line_no),
        step_mm=step_mm,
        angles_um_per_m=_parse_angles(tokens[4:], line_no),
    )
    return profile


def parse_file(path: Union[str, Path], geometry: GridGeometry,
               point_tolerance: int = 1) -> List[Profile]:
    """Прочитать все профили из файла.

    Валидирует имена узлов и геометрию профилей; расхождение фактического
    числа точек с расчётным в пределах ``±point_tolerance`` даёт
    предупреждение, большее — ошибку разбора.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Файл измерений не найден: {p}")

    profiles: List[Profile] = []
    seen_pairs: set = set()
    with p.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            try:
                prof = parse_line(line, line_no)
                geometry.node_index(prof.start)
                geometry.node_index(prof.end)
            except (ParseError, ValueError) as exc:
                if isinstance(exc, ParseError):
                    raise
                raise ParseError(line_no, str(exc)) from exc
            if not geometry.is_valid_profile(prof.start, prof.end):
                raise ParseError(
                    line_no,
                    f"профиль {prof.start}->{prof.end} не является линией сетки "
                    f"или диагональю")
            pair = (prof.start, prof.end)
            if pair in seen_pairs:
                raise ParseError(line_no, f"дублирующийся профиль {pair[0]}->{pair[1]}")
            seen_pairs.add(pair)

            expected = geometry.expected_points(prof.start, prof.end)
            actual = prof.num_points
            if abs(actual - expected) > point_tolerance:
                raise ParseError(
                    line_no,
                    f"профиль {prof.start}->{prof.end}: ожидалось {expected} "
                    f"точек (±{point_tolerance}), получено {actual}")
            if actual != expected:
                logger.warning(
                    "Строка %d: профиль %s->%s содержит %d точек вместо "
                    "расчётных %d (допуск ±%d) — шаг измерения будет "
                    "масштабирован по хорде", line_no, prof.start, prof.end,
                    actual, expected, point_tolerance)
            profiles.append(prof)

    if not profiles:
        raise ParseError(0, "файл не содержит ни одного профиля")
    logger.info("Считано профилей: %d из %s", len(profiles), p)
    return profiles
