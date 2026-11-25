"""Юнит-тесты парсера входного файла профилей (модуль ``parser``)."""

from __future__ import annotations

import logging
import textwrap
from datetime import datetime
from pathlib import Path

import pytest

from levelness.grid import GridGeometry
from levelness.parser import ParseError, Profile, parse_file, parse_line


# ------------------------------------------------------------------ parse_line
class TestParseLine:
    def test_reference_example_from_spec(self):
        prof = parse_line("A0 A6 2026-11-18T14:32 160 "
                          "0.5 0.4 0.3 0.35 0.42 0.5", 1)
        assert prof.start == "A0" and prof.end == "A6"
        assert prof.timestamp == datetime(2026, 11, 18, 14, 32)
        assert prof.step_mm == pytest.approx(160.0)
        assert prof.num_points == 6
        assert prof.angles_um_per_m[3] == pytest.approx(0.35)

    def test_tab_separated_and_comma_array(self):
        line = "B0\tB6\t2026-11-18T14:32\t160\t0.5,0.4,0.3,0.35,0.42,0.5"
        prof = parse_line(line, 2)
        assert prof.start == "B0"
        assert prof.num_points == 6
        assert prof.angles_um_per_m[0] == pytest.approx(0.5)

    def test_lowercase_node_names_uppercased(self):
        prof = parse_line("a0 a6 2026-11-18T14:32 160 0.1 0.2 0.3 0.4 0.5", 3)
        assert (prof.start, prof.end) == ("A0", "A6")

    def test_height_increments_formula(self):
        # Δh_k ≈ step_mm · θ_k  в мкм (θ в мкм/м, множитель 1e-6 сокращается)
        prof = parse_line("A0 A6 2026-11-18T14:32 160 0.5 0.4", 4)
        dh = prof.height_increments_um()
        assert dh[0] == pytest.approx(160 * 0.5)
        assert dh[1] == pytest.approx(160 * 0.4)

    def test_cumulative_heights(self):
        prof = parse_line("A0 A6 2026-11-18T14:32 160 0.5 0.4", 5)
        cum = prof.cumulative_heights_um()
        assert cum == [pytest.approx(0.0), pytest.approx(80.0),
                       pytest.approx(144.0)]

    @pytest.mark.parametrize("line,msg_part", [
        ("A0 A6 2026-11-18T14:32", "ожидались поля"),
        ("A0 A6 bad-time 160 0.1", "не удалось разобрать время"),
        ("A0 A6 2026-11-18T14:32 xx 0.1", "некорректный шаг"),
        ("A0 A6 2026-11-18T14:32 -5 0.1", "должен быть > 0"),
        ("A0 A6 2026-11-18T14:32 160 abc", "некорректное значение угла"),
        ("A0 A6 2026-11-18T14:32 160 ,", "пустой массив углов"),
    ])
    def test_errors_mention_line_number(self, line, msg_part):
        with pytest.raises(ParseError) as exc_info:
            parse_line(line, 42)
        assert exc_info.value.line_no == 42
        assert "Строка 42" in str(exc_info.value)
        assert msg_part in str(exc_info.value)


# ------------------------------------------------------------------- parse_file
def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "input.txt"
    p.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return p


ROW_OK = "A0 A6 2026-11-18T14:32 240 0.1 0.2 0.3\n"   # 3 точки при ожидании 5 → ±2? нет
LINE_5 = "A0 A6 2026-11-18T14:32 160 0.1 0.2 0.3 0.4 0.5\n"  # ровно 5 точек


class TestParseFile:
    def test_skips_comments_and_blank_lines(self, geometry: GridGeometry,
                                            tmp_path: Path):
        p = _write(tmp_path, f"""\
            # комментарий
            ; ещё комментарий

            {LINE_5}
            """)
        profiles = parse_file(p, geometry)
        assert len(profiles) == 1
        assert profiles[0].num_points == 5

    def test_exact_expected_count_ok(self, geometry: GridGeometry,
                                     tmp_path: Path):
        profiles = parse_file(_write(tmp_path, LINE_5), geometry)
        assert profiles[0].num_points == geometry.expected_points("A0", "A6")

    def test_off_by_one_is_warning_not_error(self, geometry: GridGeometry,
                                             tmp_path: Path, caplog):
        six = LINE_5.strip() + " 0.6\n"      # 6 точек вместо 5 — допуск ±1
        with caplog.at_level(logging.WARNING, logger="levelness.parser"):
            profiles = parse_file(_write(tmp_path, six), geometry,
                                  point_tolerance=1)
        assert len(profiles) == 1
        assert any("вместо расчётных" in rec.getMessage()
                   for rec in caplog.records)

    def test_off_by_two_is_error(self, geometry: GridGeometry, tmp_path: Path):
        seven = LINE_5.strip() + " 0.6 0.7\n"
        with pytest.raises(ParseError, match="ожидалось 5"):
            parse_file(_write(tmp_path, seven), geometry, point_tolerance=1)

    def test_unknown_node_reports_line_number(self, geometry: GridGeometry,
                                              tmp_path: Path):
        bad = "A0 H9 2026-11-18T14:32 160 0.1 0.2 0.3 0.4 0.5\n"
        with pytest.raises(ParseError) as exc_info:
            parse_file(_write(tmp_path, bad), geometry)
        assert "Строка 1" in str(exc_info.value)

    def test_non_grid_profile_rejected(self, geometry: GridGeometry,
                                       tmp_path: Path):
        bad = "A0 B6 2026-11-18T14:32 160 0.1 0.2 0.3 0.4 0.5\n"
        with pytest.raises(ParseError, match="линии сетки или диагонали"):
            parse_file(_write(tmp_path, bad), geometry)

    def test_duplicate_profile_rejected(self, geometry: GridGeometry,
                                        tmp_path: Path):
        with pytest.raises(ParseError, match="дублирующийся"):
            parse_file(_write(tmp_path, LINE_5 + LINE_5), geometry)

    def test_empty_file_raises(self, geometry: GridGeometry, tmp_path: Path):
        with pytest.raises(ParseError, match="ни одного профиля"):
            parse_file(_write(tmp_path, "# только комментарий\n\n"), geometry)

    def test_missing_file_raises(self, geometry: GridGeometry,
                                 tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            parse_file(tmp_path / "nope.txt", geometry)

    def test_example_input_parses_fully(self, geometry: GridGeometry,
                                        example_file: Path):
        profiles = parse_file(example_file, geometry)
        assert len(profiles) == 16                      # 7 строк + 7 столбцов + 2 диагонали
        kinds = {(p.start, p.end) for p in profiles}
        assert ("A0", "G6") in kinds and ("G0", "A6") in kinds
        for pr in profiles:
            expected = geometry.expected_points(pr.start, pr.end)
            assert abs(pr.num_points - expected) <= 1
