"""Формирование результатов: консоль, PNG/PDF-графика, CSV/JSON/HTML-экспорт."""

from __future__ import annotations

import html
import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import matplotlib

matplotlib.use("Agg")  # headless-safe рендеринг
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config, TOLERANCES_UM
from .grid import node_name
from .parser import Profile
from .solver import SolveResult, deviation_field

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
#  Консольная сводка
# --------------------------------------------------------------------------
def _fmt_matrix(mat: np.ndarray, names: List[List[str]]) -> str:
    col_w = max(8, *(len(f"{v:8.1f}") for row in mat for v in row))
    lines = []
    header = "      " + "".join(f"{'X' + str(i):>{col_w}}" for i in range(mat.shape[1]))
    lines.append(header)
    for r, name_row in enumerate(names):
        row_lbl = name_row[0][0] if name_row else ""
        cells = "".join(f"{v:>{col_w}.1f}" for v in mat[r])
        lines.append(f"{row_lbl:>4}  {cells}")
    return "\n".join(lines)


def summarize(result: SolveResult, config: Config,
              profiles: List[Profile]) -> Dict[str, Any]:
    """Словарь сводки (используется и для консоли, и для JSON/HTML)."""
    tol = config.tolerance_um()
    f = result.flatness_um
    verdict = "СООТВЕТСТВУЕТ" if f <= tol else "НЕ СООТВЕТСТВУЕТ"
    rows = [ln for ln in result.lines if ln.kind == "row"]
    cols = [ln for ln in result.lines if ln.kind == "column"]
    diags = [ln for ln in result.lines if ln.kind == "diagonal"]
    summary: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "plate": {
            "length_mm": config.geometry.length_mm,
            "width_mm": config.geometry.width_mm,
            "material": config.material,
            "precision_class": config.precision_class,
        },
        "profiles_count": len(profiles),
        "nodes": f"{config.geometry.nx}x{config.geometry.ny}",
        "flatness_um": round(f, 2),
        "tolerance_um": tol,
        "verdict": verdict,
        "network_rms_um": round(result.network_rms_um, 3),
        "plane": {
            "a_um_per_mm_x": round(result.plane_coefficients[0], 6),
            "b_um_per_mm_y": round(result.plane_coefficients[1], 6),
            "c_um": round(result.plane_coefficients[2], 3),
        },
        "skew_direction_deg": round(result.skew_direction_deg, 1),
        "skew_slope_um_per_m": round(result.skew_slope_um_per_m, 2),
        "straightness": {
            "rows_max_um": round(max((l.straightness_um for l in rows), default=0), 2),
            "columns_max_um": round(max((l.straightness_um for l in cols), default=0), 2),
            "diagonals": {l.name: round(l.straightness_um, 2) for l in diags},
        },
        "all_classes_tolerances_um": {
            cls: _interp_tol(cls, config) for cls in TOLERANCES_UM
        },
    }
    return summary


def _interp_tol(precision_class: str, config: Config) -> int:
    tmp = Config(geometry=config.geometry, precision_class=precision_class,
                 material=config.material,
                 granite_tolerance_factor=config.granite_tolerance_factor)
    return tmp.tolerance_um()


def print_summary(result: SolveResult, config: Config,
                  profiles: List[Profile]) -> Dict[str, Any]:
    """Печать таблицы узлов и сводки в консоль; возвращает словарь сводки."""
    g = result.geometry
    heights_top_down = result.node_heights[::-1, :]   # строка G сверху
    names = g.name_matrix()
    summary = summarize(result, config, profiles)

    print("=" * 72)
    print("ПЛОСКОСТНОСТЬ ПОВЕРОЧНОЙ ПЛИТЫ — СВОДКА")
    print("=" * 72)
    print(f"Плита: {g.length_mm:.0f} x {g.width_mm:.0f} мм, материал: "
          f"{config.material}, класс: {config.precision_class}")
    print(f"Сетка: {g.nx}x{g.ny} узлов, шаг X={g.step_x_mm:.1f} мм, "
          f"шаг Y={g.step_y_mm:.1f} мм, отступ {g.indent_mm:.0f} мм")
    print(f"Профилей обработано: {len(profiles)}")
    print("-" * 72)
    print("Высоты узлов, мкм (относительно среднего):")
    print(_fmt_matrix(heights_top_down, names))
    print("-" * 72)
    print(f"Плоскостность F           = {summary['flatness_um']:.2f} мкм")
    print(f"Допуск (класс {config.precision_class})       = {summary['tolerance_um']} мкм")
    print(f"Вывод                     = {summary['verdict']}")
    print(f"Направление перекоса      = {summary['skew_direction_deg']:.1f}° "
          f"(от оси X, против ч.с.)")
    print(f"Уклон наибольший          = {summary['skew_slope_um_per_m']:.2f} мкм/м")
    print(f"НСВ невязок сети          = {summary['network_rms_um']:.3f} мкм")
    st = summary["straightness"]
    print(f"Прямолинейность линий     : max строк {st['rows_max_um']} мкм, "
          f"max столбцов {st['columns_max_um']} мкм")
    for name, val in st["diagonals"].items():
        print(f"  диагональ {name:<10}: {val} мкм")
    print("=" * 72)
    return summary


# --------------------------------------------------------------------------
#  Графика
# --------------------------------------------------------------------------
def plot_deviation_map(result: SolveResult, out_dir: Path) -> List[Path]:
    """Карта отклонений: heatmap + contour, сохранение PNG и PDF."""
    X, Y, D = deviation_field(result)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    hm = ax.pcolormesh(X, Y, D, cmap="RdYlBu_r", shading="auto")
    cs = ax.contour(X, Y, D, levels=12, colors="k", linewidths=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    g = result.geometry
    gx = np.array(g.xs()); gy = np.array(g.ys())
    ax.plot(*np.meshgrid(gx, gy), "k.", ms=3, alpha=0.5)
    for jj, row in enumerate(result.residuals):
        for ii, v in enumerate(row):
            ax.annotate(f"{v:+.1f}", (gx[ii], gy[jj]), fontsize=6,
                        ha="center", va="center",
                        xytext=(0, 5), textcoords="offset points")
    fig.colorbar(hm, ax=ax, label="Отклонение от плоскости МНК, мкм")
    ax.set_xlabel("X, мм")
    ax.set_ylabel("Y, мм")
    ax.set_title(f"Карта отклонений от аппроксимирующей плоскости "
                 f"(F = {result.flatness_um:.1f} мкм)")
    ax.set_aspect("equal")
    paths = []
    for ext in ("png", "pdf"):
        p = out_dir / f"deviation_map.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    logger.info("Карта отклонений сохранена: %s", paths)
    return paths


def plot_line_profiles(result: SolveResult, out_dir: Path) -> List[Path]:
    """Графики профилей всех линий сетки и диагоналей (отклонения от прямой)."""
    groups = {"row": "Линии сетки (строки, const Y)",
              "column": "Линии сетки (столбцы, const X)",
              "diagonal": "Диагонали"}
    paths = []
    for kind, title in groups.items():
        lines = [ln for ln in result.lines if ln.kind == kind]
        if not lines:
            continue
        ncol = 2 if len(lines) > 1 else 1
        nrow = int(np.ceil(len(lines) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.6 * nrow),
                                 squeeze=False, sharex=False)
        for ax, ln in zip(axes.ravel(), lines):
            ax.plot(ln.distance_mm, ln.deviations_um, lw=1.2,
                    label=f"F_прям = {ln.straightness_um:.1f} мкм")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(ln.name, fontsize=9)
            ax.set_xlabel("дистанция, мм", fontsize=7)
            ax.set_ylabel("откл., мкм", fontsize=7)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        for ax in axes.ravel()[len(lines):]:
            ax.axis("off")
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        for ext in ("png", "pdf"):
            p = out_dir / f"profiles_{kind}.{ext}"
            fig.savefig(p, dpi=160, bbox_inches="tight")
            paths.append(p)
        plt.close(fig)
    logger.info("Профили линий сохранены: %d файлов", len(paths))
    return paths


# --------------------------------------------------------------------------
#  Экспорт данных
# --------------------------------------------------------------------------
def export_csv(result: SolveResult, config: Config, out_dir: Path) -> List[Path]:
    """Экспорт: узлы (heights.csv), линии (lines.csv), профильные данные."""
    g = result.geometry
    paths = []

    # таблица высот узлов (строки сверху вниз, как на плите)
    rows = []
    for r in range(g.ny):
        j = g.ny - 1 - r
        for i in range(g.nx):
            x, y = g.node_xy(i, j)
            rows.append({
                "node": node_name(i, j), "x_mm": round(x, 1), "y_mm": round(y, 1),
                "height_um": round(float(result.node_heights[j, i]), 3),
                "residual_plane_um": round(float(result.residuals[j, i]), 3),
            })
    df_nodes = pd.DataFrame(rows)
    p = out_dir / "node_heights.csv"
    df_nodes.to_csv(p, index=False, encoding="utf-8-sig", decimal=",", sep=";")
    paths.append(p)

    rows = []
    for ln in result.lines:
        for d, h, dev in zip(ln.distance_mm, ln.heights_um, ln.deviations_um):
            rows.append({"line": ln.name, "kind": ln.kind,
                         "distance_mm": round(d, 2), "height_um": round(h, 3),
                         "deviation_um": round(dev, 3)})
    p = out_dir / "line_profiles.csv"
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig",
                              decimal=",", sep=";")
    paths.append(p)
    logger.info("CSV сохранены: %s", [str(p) for p in paths])
    return paths


def export_json(result: SolveResult, config: Config, profiles: List[Profile],
                summary: Dict[str, Any], out_dir: Path) -> Path:
    """Полный JSON-экспорт результатов."""
    g = result.geometry
    data = {
        "summary": summary,
        "config": config.to_dict(),
        "node_heights_um": {
            node_name(i, j): round(float(result.node_heights[j, i]), 3)
            for j in range(g.ny) for i in range(g.nx)
        },
        "plane_residuals_um": {
            node_name(i, j): round(float(result.residuals[j, i]), 3)
            for j in range(g.ny) for i in range(g.nx)
        },
        "lines": [{"name": ln.name, "kind": ln.kind,
                   "straightness_um": round(ln.straightness_um, 3),
                   "distance_mm": [round(v, 2) for v in ln.distance_mm],
                   "deviations_um": [round(v, 3) for v in ln.deviations_um]}
                  for ln in result.lines],
        "profiles": [{"start": pr.start, "end": pr.end,
                      "timestamp": pr.timestamp.isoformat(),
                      "step_mm": pr.step_mm,
                      "angles_um_per_m": pr.angles_um_per_m}
                     for pr in profiles],
    }
    p = out_dir / "results.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("JSON сохранён: %s", p)
    return p


_HTML_TMPL = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>Отчёт о плоскостности плиты</title>
<style>
 body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 2em; color:#222 }}
 h1 {{ border-bottom: 3px solid #345; padding-bottom:.2em }}
 table {{ border-collapse: collapse; margin: 1em 0 }}
 th, td {{ border: 1px solid #999; padding: 4px 10px; text-align: right;
          font-variant-numeric: tabular-nums }}
 th {{ background:#345; color:#fff }}
 td.node {{ background:#eef; font-weight:bold }}
 .ok {{ color:#070; font-weight:bold }} .fail {{ color:#b00; font-weight:bold }}
 img {{ max-width: 100%; border: 1px solid #ccc; margin: .5em 0 }}
 .kv td:first-child {{ text-align:left; font-weight:bold; background:#f6f6f6 }}
</style></head><body>
<h1>Контроль плоскостности поверочной плиты</h1>
<p>Сформировано: {generated}</p>
<h2>Исходные данные</h2>
<table class="kv">{config_rows}</table>
<h2>Сводка</h2>
<table class="kv">{summary_rows}</table>
<h2>Высоты узлов сети, мкм</h2>
<table>{height_table}</table>
<h2>Прямолинейность линий и диагоналей</h2>
<table><tr><th>Линия</th><th>Тип</th><th>Прямолинейность, мкм</th></tr>
{lines_rows}</table>
<h2>Графики</h2>
{images}
</body></html>
"""


def export_html(result: SolveResult, config: Config, profiles: List[Profile],
                summary: Dict[str, Any], out_dir: Path,
                images: List[Path]) -> Path:
    """HTML-отчёт с таблицами и встроенными ссылками на графики."""
    g = result.geometry

    cfg_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in [
            ("Размер плиты, мм", f"{g.length_mm:.0f} × {g.width_mm:.0f}"),
            ("Отступ точек от края, мм", f"{g.indent_mm:.0f}"),
            ("Сетка", f"{g.nx}×{g.ny} узлов ({g.n_intervals}×{g.m_intervals} интервалов)"),
            ("Шаг сетки X/Y, мм", f"{g.step_x_mm:.1f} / {g.step_y_mm:.1f}"),
            ("Материал", config.material),
            ("Класс точности", config.precision_class),
            ("Профилей измерено", len(profiles)),
        ])

    ok_cls = "ok" if summary["verdict"].startswith("СООТ") else "fail"
    sum_rows = "".join(
        f'<tr><td>{html.escape(k)}</td>'
        f'<td class="{ok_cls if k == "Вердикт" else ""}">'
        f'{html.escape(str(v))}</td></tr>'
        for k, v in [
            ("Плоскостность F, мкм", summary["flatness_um"]),
            ("Допуск, мкм", summary["tolerance_um"]),
            ("Вердикт", summary["verdict"]),
            ("Направление перекоса, °", summary["skew_direction_deg"]),
            ("Уклон наибольший, мкм/м", summary["skew_slope_um_per_m"]),
            ("НСВ невязок сети, мкм", summary["network_rms_um"]),
        ])

    names = g.name_matrix()
    top_down = result.node_heights[::-1, :]
    ht = "<tr><th>Y\\X</th>" + "".join(f"<th>{i}</th>" for i in range(g.nx)) + "</tr>"
    for r, name_row in enumerate(names):
        ht += f'<tr><td class="node">{name_row[0]}</td>' + "".join(
            f"<td>{v:+.1f}</td>" for v in top_down[r]) + "</tr>"

    lr = "".join(
        f"<tr><td>{ln.name}</td><td>{ln.kind}</td>"
        f"<td>{ln.straightness_um:.2f}</td></tr>" for ln in result.lines)

    imgs = "".join(f'<img src="{html.escape(Path(ip).name)}" alt="{ip}">'
                   for ip in images)

    page = _HTML_TMPL.format(generated=summary["generated_at"],
                             config_rows=cfg_rows, summary_rows=sum_rows,
                             height_table=ht, lines_rows=lr, images=imgs)
    p = out_dir / "report.html"
    p.write_text(page, encoding="utf-8")
    logger.info("HTML-отчёт сохранён: %s", p)
    return p


def write_all(result: SolveResult, config: Config, profiles: List[Profile],
              out_dir: Union[str, Path]) -> Dict[str, Any]:
    """Полный вывод: консоль + графика + CSV/JSON/HTML. Возвращает сводку."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = print_summary(result, config, profiles)
    images = plot_deviation_map(result, out) + plot_line_profiles(result, out)
    export_csv(result, config, out)
    export_json(result, config, profiles, summary, out)
    export_html(result, config, profiles, summary, out, images)
    return summary
