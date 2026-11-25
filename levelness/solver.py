"""Математика обработки: МНК-восстановление узлов, плоскостность, прямолинейность.

Модель сети
-----------
Каждый профиль ``start -> end`` из ``K`` угловых отсчётов θ (мкм/м) даёт
``K+1`` равных хордовых участка длиной ``L/(K+1)``; перепад высоты на участке::

    Δh = L_chord_mm / (K+1) * θ  [мкм]      (т.к. θ мкм/м = 1e-6 рад)

Накопленные высоты вдоль профиля (h_end − h_start в узлах и относительные
высоты во внутренних точках) линейны по высотам узлов сети, поэтому вся
замкнутая сеть описывается системой ``B x = d``, где ``x`` — вектор высот
узлов (мкм), ``B`` — матрица инцидентности с весами, ``d`` — правые части.
Система переопределена и ранжирована на 1 (высота определяется с точностью
до константы); решение — метод наименьших квадратов через псевдообращение
(``numpy.linalg.lstsq``), с фиксацией нулевого уровня средним арифметическим.

Плоскостность
-------------
Аппроксимирующая плоскость ``z = a·x + b·y + c`` строится МНК по высотам
всех узлов; плоскостность ``F = max(residual) − min(residual)`` (по ISO 8512 /
DIN 876 размах относительно плоскости наименьших квадратов).

Прямолинейность
---------------
Для линии (или диагонали) — интерполированный равномерный профиль высот
между узлами; прямолинейность = размах отклонений от прямой МНК на этом
профиле.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .grid import GridGeometry, node_name
from .parser import Profile

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
#  Профиль => линейные ограничения на высоты узлов
# --------------------------------------------------------------------------
def profile_rows(profile: Profile, geometry: GridGeometry
                 ) -> Tuple[List[np.ndarray], List[float]]:
    """Строки системы ``B x = d``, создаваемые одним профилем.

    Возвращает список пар коэффициентов (вектор длины ``num_nodes``) и
    правых частей:

    * одна «жёсткая» строка — разность высот конечных узлов
      ``h_end − h_start = Σ Δh_k`` (вес 1);
    * по одной строке на каждую внутреннюю позицию профиля: если позиция
      приходится ровно на узел сети, привязка идёт к этому узлу; иначе
      высота позиции приравнивается к линейной интерполяции между концами
      профиля.  Строки имеют вес ``w_soft`` (< 1): они несут информацию о
      локальной форме (волнистости) и согласуют профили с разными базами,
      но не переопределяют глобальную геометрию замкнутой сети.
    """
    n = geometry.num_nodes
    i0, j0 = geometry.node_index(profile.start)
    i1, j1 = geometry.node_index(profile.end)
    idx0, idx1 = geometry.flat_index(i0, j0), geometry.flat_index(i1, j1)

    k = profile.num_points
    chord = geometry.chord_length_mm(profile.start, profile.end)
    # Если фактическое число точек отличается от расчётного (в пределах
    # допуска парсера), база пересчитывается по хорде: K+1 равных участков.
    seg = chord / (k + 1)                       # длина участка, мм
    if abs(seg - profile.step_mm) > 1e-6:
        logger.debug("Профиль %s->%s: шаг %.2f мм скорректирован до %.2f мм "
                     "по хорде", profile.start, profile.end,
                     profile.step_mm, seg)
    dh = [seg * theta for theta in profile.angles_um_per_m]   # перепады, мкм
    cum = np.cumsum(dh)                         # накопленные высоты, мкм
    total = float(cum[-1]) if k else 0.0

    rows: List[np.ndarray] = []
    rhs: List[float] = []

    # жёсткое условие по концам
    hard = np.zeros(n)
    hard[idx1] = 1.0
    hard[idx0] = -1.0
    rows.append(hard)
    rhs.append(total)

    # условия по внутренним позициям профиля (позиция t — после t-го Δh).
    # Накопленная высота позиции t относится к концу участка t, т.е. к доле
    # пути frac = t / (K + 1) (профиль измеряется «встык»: K отсчётов дают
    # K+1 равных участка).  Строки имеют вес w_soft (< 1): они несут
    # информацию о локальной форме (волнистости) и согласуют профили с
    # разными базами, но не переопределяют глобальную геометрию сети.
    w_soft = 0.5
    for t in range(1, k + 1):               # внутренние позиции: t < K+1
        frac = t / (k + 1)                  # доля пути start->end
        fi, fj = i0 + (i1 - i0) * frac, j0 + (j1 - j0) * frac
        row = np.zeros(n)
        if fi.is_integer() and fj.is_integer():  # точно в узле сети
            row[geometry.flat_index(int(fi), int(fj))] = w_soft
            rhs.append(w_soft * float(cum[t - 1]))
        else:                                    # линейная интерполяция концов:
            row[idx1] = w_soft * frac            # h(t) = h0 + frac*(h1 - h0)
            row[idx0] = w_soft * (frac - 1.0)
            rhs.append(w_soft * (frac - 1.0) * float(cum[t - 1]))
        rows.append(row)

    return rows, rhs


def build_network(profiles: List[Profile], geometry: GridGeometry
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Собрать полную систему ``B x = d`` по замкнутой сети профилей."""
    rows: List[np.ndarray] = []
    rhs: List[float] = []
    for prof in profiles:
        r, d = profile_rows(prof, geometry)
        rows.extend(r)
        rhs.extend(d)
    B = np.vstack(rows)
    d = np.array(rhs, dtype=float)
    logger.debug("Матрица сети: %s, ранг ≈ %d", B.shape, np.linalg.matrix_rank(B))
    return B, d


def solve_heights(profiles: List[Profile], geometry: GridGeometry
                  ) -> np.ndarray:
    """МНК-восстановление высот всех узлов сети, мкм (среднее = 0)."""
    B, d = build_network(profiles, geometry)
    x, *_ = np.linalg.lstsq(B, d, rcond=None)
    x = x - x.mean()                            # фиксация нулевого уровня
    return x.reshape(geometry.ny, geometry.nx)  # [строка Y снизу вверх][X]


# --------------------------------------------------------------------------
#  Результаты
# --------------------------------------------------------------------------
@dataclass
class LineResult:
    """Прямолинейность одной линии/диагонали."""

    name: str                                   # 'A0-A6', 'A0-G6' ...
    kind: str                                   # 'row' | 'column' | 'diagonal'
    distance_mm: List[float]                    # координата вдоль линии
    heights_um: List[float]                     # интерполированный профиль
    deviations_um: List[float]                  # отклонения от прямой МНК
    straightness_um: float                      # размах отклонений

    @property
    def endpoints(self) -> Tuple[str, str]:
        a, b = self.name.split("-")
        return a, b


@dataclass
class SolveResult:
    """Итоговый набор результатов расчёта."""

    geometry: GridGeometry
    node_heights: np.ndarray                    # [ny][nx], мкм, среднее 0
    plane_coefficients: Tuple[float, float, float]   # a, b, c: z = a x + b y + c
    residuals: np.ndarray                       # отклонения узлов от плоскости
    flatness_um: float
    skew_direction_deg: float                   # направление перекоса (град.)
    skew_slope_um_per_m: float                  # уклон наибольшего наклона
    lines: List[LineResult] = field(default_factory=list)
    network_rms_um: float = 0.0                 # НСВ невязок сети

    def grid_line_names(self) -> List[Tuple[str, str, str]]:
        out = []
        g = self.geometry
        for j in range(g.ny):
            out.append((node_name(0, j), node_name(g.nx - 1, j), "row"))
        for i in range(g.nx):
            out.append((node_name(i, 0), node_name(i, g.ny - 1), "column"))
        return out


def fit_plane(heights: np.ndarray, geometry: GridGeometry
              ) -> Tuple[Tuple[float, float, float], np.ndarray, float]:
    """Аппроксимация плоскости МНК; возвращает (a,b,c), остатки, размах."""
    xs = np.array(geometry.xs())
    ys = np.array(geometry.ys())
    X, Y = np.meshgrid(xs, ys)                  # heights[j][i] ↔ Y[j], X[i]
    A = np.column_stack([X.ravel(), Y.ravel(), np.ones(X.size)])
    z = heights.ravel()
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    resid = z - A @ coef
    a, b, c = (float(v) for v in coef)
    return (a, b, c), resid.reshape(heights.shape), float(resid.max() - resid.min())


def line_profile(heights: np.ndarray, geometry: GridGeometry,
                 start: str, end: str, samples_per_step: int = 4
                 ) -> LineResult:
    """Равномерно интерполированный профиль между двумя узлами + прямолинейность."""
    i0, j0 = geometry.node_index(start)
    i1, j1 = geometry.node_index(end)
    x0, y0 = geometry.node_xy(i0, j0)
    steps = geometry.grid_steps(start, end)
    n_samples = steps * samples_per_step + 1
    xs, ys, hs = [], [], []
    for s in range(n_samples):
        t = s / (n_samples - 1)
        fi, fj = i0 + (i1 - i0) * t, j0 + (j1 - j0) * t
        bi, bj = min(int(fi), geometry.nx - 2), min(int(fj), geometry.ny - 2)
        ti, tj = fi - bi, fj - bj               # билинейная интерполяция
        h = ((1 - ti) * (1 - tj) * heights[bj, bi]
             + ti * (1 - tj) * heights[bj, bi + 1]
             + (1 - ti) * tj * heights[bj + 1, bi]
             + ti * tj * heights[bj + 1, bi + 1])
        x = geometry.x(i0) + (geometry.x(geometry.nx - 1) - geometry.x(0)) * fi \
            if i1 != i0 else geometry.x(i0)
        y = geometry.y(j0) + (geometry.y(geometry.ny - 1) - geometry.y(j0)) * fj \
            if j1 != j0 else geometry.y(j0)
        dist = math.hypot(x - x0, y - y0)
        xs.append(x); ys.append(y); hs.append(float(h))
    dist_arr = np.array([math.hypot(xx - x0, yy - y0) for xx, yy in zip(xs, ys)])
    # прямая МНК вдоль профиля
    A = np.column_stack([dist_arr, np.ones_like(dist_arr)])
    coef, *_ = np.linalg.lstsq(A, np.array(hs), rcond=None)
    dev = np.array(hs) - A @ coef
    di, dj = abs(i1 - i0), abs(j1 - j0)
    kind = "row" if dj == 0 else ("column" if di == 0 else "diagonal")
    return LineResult(
        name=f"{start.upper()}-{end.upper()}",
        kind=kind,
        distance_mm=[float(v) for v in dist_arr],
        heights_um=[float(v) for v in hs],
        deviations_um=[float(v) for v in dev],
        straightness_um=float(dev.max() - dev.min()),
    )


def main_diagonals(geometry: GridGeometry) -> List[Tuple[str, str]]:
    """Две главные диагонали сетки."""
    nx, ny = geometry.nx, geometry.ny
    return [(node_name(0, 0), node_name(nx - 1, ny - 1)),
            (node_name(nx - 1, 0), node_name(0, ny - 1))]


def process(profiles: List[Profile], geometry: GridGeometry,
            include_all_diagonals: bool = False) -> SolveResult:
    """Полный цикл: сеть → высоты узлов → плоскость → прямолинейность линий."""
    heights = solve_heights(profiles, geometry)

    # Невязки сети (для контроля качества измерений)
    B, d = build_network(profiles, geometry)
    resid_net = B @ heights.ravel() - d
    rms = float(np.sqrt(np.mean(resid_net ** 2)))

    (a, b, c), residuals, flatness = fit_plane(heights, geometry)
    skew_dir = math.degrees(math.atan2(b, a)) % 360.0
    slope_um_per_m = math.hypot(a, b) * 1e3     # a,b в мкм/мм → мкм/м

    lines: List[LineResult] = []
    for s, e, _kind in [(node_name(0, j), node_name(geometry.nx - 1, j), "row")
                        for j in range(geometry.ny)] + \
                       [(node_name(i, 0), node_name(i, geometry.ny - 1), "column")
                        for i in range(geometry.nx)]:
        lines.append(line_profile(heights, geometry, s, e))
    diag_pairs = main_diagonals(geometry)
    if include_all_diagonals:  # все короткие диагонали клеток сети
        for i in range(geometry.nx - 1):
            for j in range(geometry.ny - 1):
                diag_pairs += [(node_name(i, j), node_name(i + 1, j + 1)),
                               (node_name(i + 1, j), node_name(i, j + 1))]
    for s, e in diag_pairs:
        lines.append(line_profile(heights, geometry, s, e))

    logger.info("Плоскостность F = %.2f мкм; НСВ невязок сети = %.2f мкм",
                flatness, rms)
    return SolveResult(
        geometry=geometry,
        node_heights=heights,
        plane_coefficients=(a, b, c),
        residuals=residuals,
        flatness_um=flatness,
        skew_direction_deg=skew_dir,
        skew_slope_um_per_m=slope_um_per_m,
        lines=lines,
        network_rms_um=rms,
    )


def deviation_field(result: SolveResult, resolution: int = 60
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Поле отклонений от аппроксимирующей плоскости на мелкой сетке (для карты)."""
    g = result.geometry
    xs = np.linspace(g.x(0), g.x(g.nx - 1), resolution)
    ys = np.linspace(g.y(0), g.y(g.ny - 1), resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    H = result.node_heights
    step_x, step_y = g.step_x_mm, g.step_y_mm
    for jj in range(Y.shape[0]):
        for ii in range(Y.shape[1]):
            fi = (X[jj, ii] - g.x(0)) / step_x
            fj = (Y[jj, ii] - g.y(0)) / step_y
            bi = min(int(fi), g.nx - 2)
            bj = min(int(fj), g.ny - 2)
            ti, tj = fi - bi, fj - bj
            Z[jj, ii] = ((1 - ti) * (1 - tj) * H[bj, bi]
                         + ti * (1 - tj) * H[bj, bi + 1]
                         + (1 - ti) * tj * H[bj + 1, bi]
                         + ti * tj * H[bj + 1, bi + 1])
    a, b, c = result.plane_coefficients
    return X, Y, Z - (a * X + b * Y + c)
