"""Юнит-тесты геометрии плиты и измерительной сетки (модуль ``grid``)."""

from __future__ import annotations

import math

import pytest

from levelness.grid import GridGeometry, node_name, parse_node_name


# ---------------------------------------------------------------- имена узлов
class TestNodeNames:
    def test_basic_pairing(self):
        assert node_name(0, 0) == "A0"          # левый нижний угол
        assert node_name(6, 0) == "A6"          # правый нижний угол
        assert node_name(0, 6) == "G0"          # левый верхний угол

    def test_roundtrip(self):
        for i in range(7):
            for j in range(7):
                assert parse_node_name(node_name(i, j)) == (i, j)

    def test_parse_case_insensitive_and_stripped(self):
        assert parse_node_name(" a6 ") == (6, 0)
        assert parse_node_name("G3") == (3, 6)

    def test_invalid_names_raise(self):
        for bad in ("", "0A", "A", "AA"):
            with pytest.raises(ValueError):
                parse_node_name(bad)

    def test_negative_index_raises(self):
        with pytest.raises(ValueError):
            node_name(-1, 0)


# ------------------------------------------------------------------ координаты
class TestCoordinates:
    def test_default_grid_values(self, geometry: GridGeometry):
        assert (geometry.nx, geometry.ny) == (7, 7)
        assert geometry.num_nodes == 49
        assert geometry.step_x_mm == pytest.approx(160.0)
        assert geometry.step_y_mm == pytest.approx((630 - 40) / 6)

    def test_corner_coordinates(self, geometry: GridGeometry):
        # A0 — левый нижний (20, 20); A6 — правый нижний (980, 20)
        assert geometry.node_xy(0, 0) == pytest.approx((20.0, 20.0))
        assert geometry.node_xy(geometry.nx - 1, 0) == pytest.approx((980.0, 20.0))
        # G0 — левый верхний (20, 610)
        assert geometry.node_xy(0, geometry.ny - 1) == pytest.approx((20.0, 610.0))

    def test_formula_x_i_indent_plus_i_step(self, geometry: GridGeometry):
        for i in range(geometry.nx):
            assert geometry.x(i) == pytest.approx(
                20 + i * (1000 - 2 * 20) / 6)
        for j in range(geometry.ny):
            assert geometry.y(j) == pytest.approx(
                20 + j * (630 - 2 * 20) / 6)

    def test_flat_index_row_major_from_bottom(self, geometry: GridGeometry):
        assert geometry.flat_index(0, 0) == 0
        assert geometry.flat_index(6, 0) == 6
        assert geometry.flat_index(0, 1) == 7
        assert geometry.flat_index(6, 6) == 48


# ------------------------------------------------------------- валидация формы
class TestValidation:
    @pytest.mark.parametrize("kwargs", [
        dict(length_mm=-5),
        dict(width_mm=0),
        dict(n_intervals=0),
        dict(indent_mm=400),           # >= min(L, W)/2
        dict(indent_mm=-1),
    ])
    def test_bad_geometry_raises(self, kwargs):
        base = dict(length_mm=1000, width_mm=630, indent_mm=20,
                    n_intervals=6, m_intervals=6)
        base.update(kwargs)
        with pytest.raises(ValueError):
            GridGeometry(**base)

    def test_out_of_grid_node_rejected(self, geometry: GridGeometry):
        with pytest.raises(ValueError):
            geometry.node_index("H0")   # строки только A..G
        with pytest.raises(ValueError):
            geometry.node_index("A7")   # столбцы только 0..6


# ------------------------------------------------------------------- профили
class TestProfiles:
    def test_expected_points_row_column_diagonal(self, geometry: GridGeometry):
        assert geometry.expected_points("A0", "A6") == 5     # N - 1
        assert geometry.expected_points("A0", "G0") == 5     # M - 1
        assert geometry.expected_points("A0", "G6") == 6     # max(N,M) - 1

    def test_chord_lengths(self, geometry: GridGeometry):
        assert geometry.chord_length_mm("A0", "A6") == pytest.approx(960.0)
        assert geometry.chord_length_mm("A0", "G0") == pytest.approx(590.0)
        assert geometry.chord_length_mm("A0", "G6") == pytest.approx(
            math.hypot(960.0, 590.0))

    def test_valid_profile_kinds(self, geometry: GridGeometry):
        assert geometry.is_valid_profile("A0", "A6")      # горизонталь
        assert geometry.is_valid_profile("A0", "G0")      # вертикаль
        assert geometry.is_valid_profile("A0", "G6")      # диагональ
        assert geometry.is_valid_profile("C2", "E4")      # короткая диагональ
        assert not geometry.is_valid_profile("A0", "B6")  # «рыбий хвост»
        assert not geometry.is_valid_profile("A0", "A0")  # нулевой профиль

    def test_along_coords_uniform(self, geometry: GridGeometry):
        pts = geometry.along_coords("A0", "A6", 5)
        assert len(pts) == 5
        assert pts[0][0] == pytest.approx(20 + 960 / 6)
        assert all(abs(p[1] - 20) < 1e-12 for p in pts)

    def test_name_matrix_top_row_is_G(self, geometry: GridGeometry):
        m = geometry.name_matrix()
        assert m[0][0].startswith("G")     # печатаем сверху вниз: G..A
        assert m[-1] == ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
