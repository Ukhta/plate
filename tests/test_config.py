"""Юнит-тесты конфигурации (модуль ``config``): JSON, допуски, CLI-оверрайды."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from levelness.config import Config, TOLERANCES_UM, VALID_CLASSES
from levelness.grid import GridGeometry


class TestDefaults:
    def test_default_geometry_matches_spec(self):
        cfg = Config()
        g = cfg.geometry
        assert (g.length_mm, g.width_mm, g.indent_mm) == (1000.0, 630.0, 20.0)
        assert (g.n_intervals, g.m_intervals) == (6, 6)

    def test_defaults_are_valid(self):
        cfg = Config()
        assert cfg.precision_class in VALID_CLASSES
        assert cfg.tolerance_um() >= 1


class TestFromDictAndFile:
    def test_from_dict_full(self):
        cfg = Config.from_dict({
            "plate": {"length_mm": 1250, "width_mm": 800, "indent_mm": 25},
            "grid": {"n_intervals": 8, "m_intervals": 5},
            "precision_class": "0",
            "material": "granite",
        })
        assert cfg.geometry.length_mm == 1250
        assert cfg.geometry.nx == 9 and cfg.geometry.ny == 6
        assert cfg.precision_class == "0"
        assert cfg.material == "granite"

    def test_load_json_file(self, tmp_path: Path):
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"plate": {"length_mm": 630, "width_mm": 400},
                                 "precision_class": "2"}), encoding="utf-8")
        cfg = Config.load(p)
        assert cfg.geometry.length_mm == 630
        assert cfg.precision_class == "2"

    def test_load_none_gives_defaults(self):
        assert Config.load(None).geometry == GridGeometry()

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            Config.load(tmp_path / "absent.json")

    def test_broken_json_raises_valueerror(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{ not json ]", encoding="utf-8")
        with pytest.raises(ValueError, match="Ошибка разбора JSON"):
            Config.load(p)

    @pytest.mark.parametrize("cls", ["00", "0", "1", "2"])
    def test_roundtrip_all_classes(self, cls: str):
        cfg = Config(precision_class=cls)
        assert Config.from_dict(cfg.to_dict()).precision_class == cls

    def test_invalid_class_rejected(self):
        with pytest.raises(ValueError, match="Класс точности"):
            Config(precision_class="3")

    def test_invalid_material_rejected(self):
        with pytest.raises(ValueError, match="material"):
            Config(material="steel")


class TestTolerances:
    def test_ordering_of_classes(self):
        g = GridGeometry()
        tols = [Config(geometry=g, precision_class=c).tolerance_um()
                for c in ("00", "0", "1", "2")]
        assert tols == sorted(tols) and len(set(tols)) == 4

    def test_table_monotonic_by_size(self):
        """Внутри класса допуск не убывает с ростом типоразмера."""
        for cls, table in TOLERANCES_UM.items():
            vals = [table[s] for s in sorted(table)]
            assert vals == sorted(vals), cls

    def test_default_plate_class1_tolerance(self):
        # диагональ сетки 960×590 ≈ 1127 мм → интерполяция между 1000 и 1250
        tol = Config(precision_class="1").tolerance_um()
        assert 11 <= tol <= 12

    def test_granite_factor_scales_tolerance(self):
        base = Config(precision_class="1").tolerance_um()
        strict = Config(precision_class="1",
                        granite_tolerance_factor=0.8).tolerance_um()
        assert strict < base


class TestOverrides:
    def test_cli_overrides_beat_file(self):
        cfg = Config().with_overrides({"length_mm": 800.0,
                                       "precision_class": "00"})
        assert cfg.geometry.length_mm == 800.0
        assert cfg.precision_class == "00"
        # остальные параметры не затронуты
        assert cfg.geometry.width_mm == 630.0

    def test_none_values_ignored(self):
        cfg = Config().with_overrides({"length_mm": None, "material": None})
        assert cfg.geometry.length_mm == 1000.0
        assert cfg.material == "cast_iron"

    def test_to_dict_contains_key_fields(self):
        d = Config().to_dict()
        assert d["plate"]["length_mm"] == 1000
        assert d["grid"]["nodes"] == "7x7"
        assert "tolerance_um" in d
