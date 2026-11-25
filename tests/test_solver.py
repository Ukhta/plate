"""Юнит-тесты МНК-восстановления и расчётных характеристик (модуль ``solver``).

Ключевой тест — восстановление синтетического профиля: из аналитической
поверхности генерируются идеальные угловые отсчёты, замкнутая сеть решается
МНК, высоты узлов сравниваются с истиной.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from levelness.grid import GridGeometry, node_name
from levelness.parser import Profile
from levelness.solver import (
    build_network,
    deviation_field,
    fit_plane,
    line_profile,
    main_diagonals,
    process,
    profile_rows,
    solve_heights,
)


# ---------------------------------------------------------------- helpers
def synth_profiles(geometry: GridGeometry,
                   z, *, pairs=None, t0="2026-11-18T14:00") -> list[Profile]:
    """Идеальные профили поверхности ``z(x_mm_rel, y_mm_rel)`` (мкм).

    Угол на участке: θ = Δh / seg_len (мкм/мм == мкм/м), участки «встык»
    по расчётному числу точек ТЗ.
    """
    if pairs is None:
        pairs = []
        for j in range(geometry.ny):
            pairs.append((node_name(0, j), node_name(geometry.nx - 1, j)))
        for i in range(geometry.nx):
            pairs.append((node_name(i, 0), node_name(i, geometry.ny - 1)))
        pairs += main_diagonals(geometry)

    out: list[Profile] = []
    for s, e in pairs:
        k = geometry.expected_points(s, e)
        chord = geometry.chord_length_mm(s, e)
        seg = chord / (k + 1)
        x0, y0 = geometry.node_xy(*geometry.node_index(s))
        x1, y1 = geometry.node_xy(*geometry.node_index(e))
        ux, uy = (x1 - x0) / chord, (y1 - y0) / chord
        angles = []
        for t in range(k):
            ax = x0 + ux * seg * t - geometry.x(0)
            ay = y0 + uy * seg * t - geometry.y(0)
            bx, by = ax + ux * seg, ay + uy * seg
            angles.append((z(bx, by) - z(ax, ay)) / seg)
        out.append(Profile(start=s, end=e, timestamp=t0, step_mm=seg,
                           angles_um_per_m=angles))
    return out


def true_z_fn(geometry: GridGeometry):
    """Наклонная плоскость + квадратичный прогиб, мкм."""
    lx = geometry.x(geometry.nx - 1) - geometry.x(0)
    ly = geometry.y(geometry.ny - 1) - geometry.y(0)

    def z(x_mm: float, y_mm: float) -> float:
        u, v = x_mm / lx, y_mm / ly
        return 2.0 * u + 3.0 * v + 8.0 * (u * u + v * v) - 6.0 * u * v
    return z


# --------------------------------------------------------------- Profile math
class TestProfileMath:
    def test_delta_h_matches_spec_formula(self):
        # Δh_k ≈ step_mm · θ_k · 1e-6 [м] = step_mm · θ_k [мкм]
        prof = Profile("A0", "A6", "2026-11-18T14:32", 160.0, [0.5, 0.4])
        assert prof.height_increments_um() == pytest.approx([80.0, 64.0])

    def test_chord_is_end_to_end(self):
        prof = Profile("A0", "A6", "2026-11-18T14:32", 160.0, [0.1] * 5)
        assert prof.chord_mm == pytest.approx(960.0)


# ------------------------------------------------------------------ network
class TestNetwork:
    def test_hard_row_encodes_end_difference(self, geometry: GridGeometry):
        prof = Profile("A0", "A6", "2026-11-18T14:32", 160.0, [0.5] * 5)
        rows, rhs = profile_rows(prof, geometry)
        hard, d0 = rows[0], rhs[0]
        assert hard[geometry.flat_index(6, 0)] == 1.0
        assert hard[geometry.flat_index(0, 0)] == -1.0
        assert d0 == pytest.approx(5 * 160 * 0.5)      # Σ Δh

    def test_matrix_shape(self, geometry: GridGeometry):
        profs = synth_profiles(geometry, lambda x, y: 0.001 * x)
        B, d = build_network(profs, geometry)
        assert B.shape[1] == geometry.num_nodes
        assert B.shape[0] == d.shape[0] >= geometry.num_nodes


# --------------------------------------------------- synthetic reconstruction
class TestSyntheticReconstruction:
    def test_plane_recovered_exactly(self, geometry: GridGeometry):
        """Линейная поверхность восстанавливается МНК без ошибок сети."""
        z = lambda x, y: 1.5 * x / 960 + 2.5 * y / 590          # мкм
        heights = solve_heights(synth_profiles(geometry, z), geometry)
        truth = np.array([[z(geometry.x(i) - geometry.x(0),
                             geometry.y(j) - geometry.y(0))
                           for i in range(geometry.nx)]
                          for j in range(geometry.ny)])
        truth -= truth.mean()
        assert np.allclose(heights, truth, atol=1e-6)

    def test_quadratic_surface_recovered(self, geometry: GridGeometry):
        """Синтетическая поверхность ТЗ восстанавливается с малой ошибкой."""
        fn = true_z_fn(geometry)
        result = process(synth_profiles(geometry, fn), geometry)
        truth = np.array([[fn(geometry.x(i) - geometry.x(0),
                               geometry.y(j) - geometry.y(0))
                            for i in range(geometry.nx)]
                          for j in range(geometry.ny)])
        truth -= truth.mean()
        err = np.abs(result.node_heights - truth).max()
        assert err < 0.5                       # мкм — узлы восстановлены
        assert result.network_rms_um < 1e-6    # сеть невязок ~ 0

    def test_mean_of_heights_is_zero(self, geometry: GridGeometry):
        heights = solve_heights(
            synth_profiles(geometry, true_z_fn(geometry)), geometry)
        assert heights.mean() == pytest.approx(0.0, abs=1e-9)

    def test_example_input_gives_small_flatness(self, geometry: GridGeometry,
                                                example_file):
        from levelness.parser import parse_file
        result = process(parse_file(example_file, geometry), geometry)
        # синтетика ТЗ: F истинной поверхности ≈ 4–5 мкм
        assert 1.0 < result.flatness_um < 10.0


# -------------------------------------------------------------------- plane
class TestPlaneFit:
    def test_plane_recovers_coefficients(self, geometry: GridGeometry):
        a, b, c = 0.01, -0.02, 5.0     # мкм/мм
        h = np.array([[a * geometry.x(i) + b * geometry.y(j) + c
                       for i in range(geometry.nx)]
                      for j in range(geometry.ny)])
        (fa, fb, fc), resid, flatness = fit_plane(h, geometry)
        assert fa == pytest.approx(a, abs=1e-9)
        assert fb == pytest.approx(b, abs=1e-9)
        assert fc == pytest.approx(c, abs=1e-9)
        assert flatness == pytest.approx(0.0, abs=1e-9)
        assert np.abs(resid).max() < 1e-9

    def test_flatness_is_range_of_residuals(self, geometry: GridGeometry):
        rng = np.random.default_rng(7)
        h = rng.normal(scale=3.0, size=(geometry.ny, geometry.nx))
        _, resid, flatness = fit_plane(h, geometry)
        assert flatness == pytest.approx(resid.max() - resid.min())


# -------------------------------------------------------------- straightness
class TestLineProfile:
    def test_straight_line_has_zero_deviation(self, geometry: GridGeometry):
        h = np.array([[0.01 * i for i in range(geometry.nx)]
                      for _ in range(geometry.ny)])
        ln = line_profile(h, geometry, "A0", "A6")
        assert ln.straightness_um == pytest.approx(0.0, abs=1e-9)
        assert ln.kind == "row"

    def test_column_and_diagonal_kinds(self, geometry: GridGeometry):
        h = np.zeros((geometry.ny, geometry.nx))
        assert line_profile(h, geometry, "A0", "G0").kind == "column"
        d = line_profile(h, geometry, "A0", "G6")
        assert d.kind == "diagonal"
        assert len(d.distance_mm) == len(d.heights_um)
        assert d.distance_mm[0] == pytest.approx(0.0)
        assert d.distance_mm[-1] == pytest.approx(
            geometry.chord_length_mm("A0", "G6"), rel=1e-9)

    def test_bump_detected_on_line(self, geometry: GridGeometry):
        h = np.zeros((geometry.ny, geometry.nx))
        h[0, 3] = 10.0                          # «горб» в середине строки A
        ln = line_profile(h, geometry, "A0", "A6")
        assert ln.straightness_um > 5.0


# ---------------------------------------------------------------- full cycle
class TestProcessResult:
    @pytest.fixture()
    def result(self, geometry: GridGeometry):
        return process(synth_profiles(geometry, true_z_fn(geometry)), geometry)

    def test_lines_complete(self, result, geometry: GridGeometry):
        kinds = [ln.kind for ln in result.lines]
        assert kinds.count("row") == geometry.ny
        assert kinds.count("column") == geometry.nx
        assert kinds.count("diagonal") == 2

    def test_skew_direction_consistency(self, result):
        a, b, _ = result.plane_coefficients
        expected = math.degrees(math.atan2(b, a)) % 360.0
        assert result.skew_direction_deg == pytest.approx(expected)
        assert result.skew_slope_um_per_m == pytest.approx(
            math.hypot(a, b) * 1e3)

    def test_all_diagonals_option(self, geometry: GridGeometry):
        res = process(synth_profiles(geometry, lambda x, y: 0.0),
                      geometry, include_all_diagonals=True)
        n_cells = (geometry.nx - 1) * (geometry.ny - 1)
        diags = [ln for ln in res.lines if ln.kind == "diagonal"]
        assert len(diags) == 2 + 2 * n_cells

    def test_deviation_field_shape_and_centering(self, result):
        X, Y, D = deviation_field(result, resolution=25)
        assert X.shape == Y.shape == D.shape == (25, 25)
        # для чистого прогиба поле конечно и связно
        assert np.isfinite(D).all()
