"""Интеграционные тесты отчёта и CLI (модули ``report``, ``cli``)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from levelness.cli import main
from levelness.config import Config
from levelness.parser import parse_file
from levelness.report import summarize, write_all
from levelness.solver import process


@pytest.fixture()
def solved(geometry, example_file):
    profiles = parse_file(example_file, geometry)
    cfg = Config(geometry=geometry, precision_class="1")
    return cfg, profiles, process(profiles, geometry)


class TestSummary:
    def test_verdict_conforms(self, solved):
        cfg, profiles, result = solved
        summary = summarize(result, cfg, profiles)
        assert summary["flatness_um"] <= summary["tolerance_um"]
        assert summary["verdict"] == "СООТВЕТСТВУЕТ"

    def test_verdict_fails_when_class_too_strict(self, solved):
        cfg, profiles, result = solved
        strict = Config(geometry=cfg.geometry, precision_class="00")
        summary = summarize(result, strict, profiles)
        assert summary["verdict"] == "НЕ СООТВЕТСТВУЕТ"

    def test_summary_fields(self, solved):
        cfg, profiles, result = solved
        s = summarize(result, cfg, profiles)
        assert s["profiles_count"] == len(profiles)
        assert s["nodes"] == "7x7"
        assert set(s["all_classes_tolerances_um"]) == {"00", "0", "1", "2"}
        assert 0 <= s["skew_direction_deg"] < 360


class TestWriteAll:
    def test_all_artifacts_created(self, solved, tmp_path: Path, capsys):
        cfg, profiles, result = solved
        out = tmp_path / "report"
        summary = write_all(result, cfg, profiles, out)
        expected = ["deviation_map.png", "deviation_map.pdf",
                    "profiles_row.png", "profiles_row.pdf",
                    "profiles_column.png", "profiles_column.pdf",
                    "profiles_diagonal.png", "profiles_diagonal.pdf",
                    "node_heights.csv", "line_profiles.csv",
                    "results.json", "report.html"]
        for name in expected:
            assert (out / name).exists(), name
        # консольная сводка напечатана
        printed = capsys.readouterr().out
        assert "ПЛОСКОСТНОСТЬ" in printed and "Вывод" in printed
        assert summary["flatness_um"] > 0

    def test_results_json_structure(self, solved, tmp_path: Path):
        cfg, profiles, result = solved
        write_all(result, cfg, profiles, tmp_path)
        data = json.loads((tmp_path / "results.json").read_text("utf-8"))
        assert set(data) >= {"summary", "config", "node_heights_um",
                             "plane_residuals_um", "lines", "profiles"}
        assert len(data["node_heights_um"]) == 49
        assert "A0" in data["node_heights_um"]

    def test_html_mentions_key_numbers(self, solved, tmp_path: Path):
        cfg, profiles, result = solved
        write_all(result, cfg, profiles, tmp_path)
        page = (tmp_path / "report.html").read_text("utf-8")
        assert "<html" in page
        assert "Контроль плоскостности" in page
        assert "deviation_map.png" in page


class TestCli:
    def test_end_to_end_exit_zero(self, example_file, tmp_path: Path,
                                  capsys):
        rc = main([str(example_file), "--out", str(tmp_path / "rep")])
        assert rc == 0
        assert (tmp_path / "rep" / "results.json").exists()
        assert "Плоскостность F" in capsys.readouterr().out

    def test_config_and_overrides(self, example_file, tmp_path: Path):
        rc = main([str(example_file), "--class", "00",
                   "--out", str(tmp_path / "rep2")])
        assert rc == 0
        data = json.loads((tmp_path / "rep2" / "results.json").read_text())
        assert data["config"]["precision_class"] == "00"
        assert data["summary"]["verdict"] == "НЕ СООТВЕТСТВУЕТ"

    def test_bad_input_returns_2(self, tmp_path: Path):
        bad = tmp_path / "bad.txt"
        bad.write_text("A0 A6 not-a-date 160 0.1\n", encoding="utf-8")
        assert main([str(bad), "--out", str(tmp_path / "r")]) == 2

    def test_missing_file_returns_2(self, tmp_path: Path):
        assert main([str(tmp_path / "ghost.txt"), "--out",
                     str(tmp_path / "r")]) == 2

    def test_all_diagonals_flag(self, example_file, tmp_path: Path):
        rc = main([str(example_file), "--out", str(tmp_path / "r3"),
                   "--all-diagonals"])
        assert rc == 0
        data = json.loads((tmp_path / "r3" / "results.json").read_text())
        diags = [ln for ln in data["lines"] if ln["kind"] == "diagonal"]
        assert len(diags) == 2 + 2 * 6 * 6
