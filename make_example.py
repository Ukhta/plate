"""Генерация example_input.txt — синтетические измерения известной поверхности.

Модель истинной высоты поверхности (мкм), u = x/L, v = y/L в долях сетки:

    z = 8(u² + v²) − 6uv + 3u + 1.5·sin(πu)·sin(πv) + 0.4·sin(3πu)

Углы θ на участках профиля = разность высот соседних позиций / длина участка.
Длины участков подобраны так, чтобы число точек совпало с расчётным по ТЗ
(N−1 для строк/столбцов, max(N,M)−1 для диагоналей).
"""
import math

from levelness.grid import GridGeometry, node_name

g = GridGeometry(1000, 630, 20, 6, 6)
Lx = g.x(g.nx - 1) - g.x(0)   # 960 мм
Ly = g.y(g.ny - 1) - g.y(0)   # 590 мм


def true_z(x_mm: float, y_mm: float) -> float:
    """Истинная высота, мкм; аргументы — координаты в мм от угла сетки."""
    u, v = x_mm / Lx, y_mm / Ly
    return (8 * (u * u + v * v) - 6 * u * v + 3 * u
            + 1.5 * math.sin(math.pi * u) * math.sin(math.pi * v)
            + 0.4 * math.sin(3 * math.pi * u))


pairs = []
for j in range(g.ny):                       # строки A..G (по X)
    pairs.append((node_name(0, j), node_name(g.nx - 1, j)))
for i in range(g.nx):                       # столбцы 0..6 (по Y)
    pairs.append((node_name(i, 0), node_name(i, g.ny - 1)))
pairs += [("A0", "G6"), ("G0", "A6")]       # главные диагонали

lines = []
for n, (s, e) in enumerate(pairs):
    k = g.expected_points(s, e)             # расчётное число точек по ТЗ
    chord = g.chord_length_mm(s, e)
    seg = chord / (k + 1)                   # длина участка, мм
    x0, y0 = g.node_xy(*g.node_index(s))
    x1, y1 = g.node_xy(*g.node_index(e))
    ux, uy = (x1 - x0) / chord, (y1 - y0) / chord
    angles = []
    for t in range(k):                      # K углов на первых K участках
        ax = x0 + ux * seg * t - g.x(0)
        ay = y0 + uy * seg * t - g.y(0)
        bx, by = ax + ux * seg, ay + uy * seg
        theta = (true_z(bx, by) - true_z(ax, ay)) / seg   # мкм/м == 1e-6 рад
        angles.append(round(theta, 4))
    ts = f"2026-11-18T{14}:{30 + n:02d}"
    lines.append(f"{s} {e} {ts} {seg:.1f} " + " ".join(str(a) for a in angles))

with open("example_input.txt", "w", encoding="utf-8") as f:
    f.write("# профиль: <старт> <финиш> <время> <шаг,мм> <углы, мкм/м>\n")
    f.write("\n".join(lines) + "\n")
print(f"Записано профилей: {len(lines)}")
