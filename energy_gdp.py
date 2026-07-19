#!/usr/bin/env python3
"""Reproduce the joint distribution of GDP and energy across the 1980-2023 panel.

Reads the two saved raw datasets in ``data/``, merges them into a single
country-year panel, characterises the joint law of real GDP per capita and
primary energy use per capita, and writes:

    energy-gdp-panel-1980-2023.csv   the merged panel (one row per country-year)
    energy-gdp-chart.svg             the log-log density chart (self-contained)

and prints the headline statistics. Deterministic: same inputs -> identical
outputs, byte for byte.

    python energy_gdp.py

Standard library only (Python 3.10+). No third-party dependencies, no network.

GDP is measured in constant 2015 US$ (real), so the 44-year span is not
distorted by inflation.

Inputs (data/):
    owid-primary-energy-per-capita.csv                          Our World in Data
        (Energy Institute Statistical Review; U.S. EIA), primary energy per
        capita, kWh/yr. Column: primary_energy_consumption_per_capita__kwh.
    worldbank-gdp-per-capita-constant-2015-usd-1980-2024.json   World Bank
        NY.GDP.PCAP.KD, GDP per capita, constant 2015 US$.
"""
import json
import csv
import math
import html
from collections import defaultdict
from pathlib import Path

Y0YEAR, Y1YEAR = 1980, 2023

HERE = Path(__file__).resolve().parent
RAW = HERE / "data"

ENERGY_CSV = RAW / "owid-primary-energy-per-capita.csv"
GDP_JSON = RAW / "worldbank-gdp-per-capita-constant-2015-usd-1980-2024.json"
OUT_CSV = HERE / "energy-gdp-panel-1980-2023.csv"
OUT_SVG = HERE / "energy-gdp-chart.svg"

# Low-energy threshold for the "empty corner": no high-income (>= $30k, real
# 2015 US$) country-year falls below this many kWh of primary energy per capita.
LOW_KWH = 10000

# World Bank aggregate/region codes to exclude so only sovereign entities remain.
AGG = {"AFE", "AFW", "ARB", "CEB", "CSS", "EAP", "EAR", "EAS", "ECA", "ECS",
       "EMU", "EUU", "FCS", "HIC", "HPC", "IBD", "IBT", "IDA", "IDB", "IDX",
       "LAC", "LCN", "LDC", "LIC", "LMC", "LMY", "LTE", "MEA", "MIC", "MNA",
       "NAC", "OED", "OSS", "PRE", "PSS", "PST", "SAS", "SSA", "SSF", "SST",
       "TEA", "TEC", "TLA", "TMN", "TSA", "TSS", "UMC", "WLD", "INX"}


def esc(s):
    return html.escape(str(s))


def compact_years(years):
    """[2001,2002,2003,2005] -> '2001-2003, 2005'"""
    ys = sorted(years)
    runs, start, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if y == prev + 1:
            prev = y
            continue
        runs.append((start, prev))
        start = prev = y
    runs.append((start, prev))
    return ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)


# --------------------------------------------------------------- data & stats
def load_panel():
    """Return list of (iso, name, year, gdp, energy) with both values present."""
    energy = {}
    with open(ENERGY_CSV) as f:
        for row in csv.DictReader(f):
            c = row["code"]
            if c and len(c) == 3:
                try:
                    energy[(c, int(row["year"]))] = float(row["primary_energy_consumption_per_capita__kwh"])
                except ValueError:
                    pass
    panel = []
    for r in json.load(open(GDP_JSON))[1]:
        iso = r["countryiso3code"]
        if r["value"] is None or iso in AGG:
            continue
        yr = int(r["date"])
        if not (Y0YEAR <= yr <= Y1YEAR):
            continue
        e = energy.get((iso, yr))
        g = float(r["value"])
        if e and g > 0 and e > 0:
            panel.append((iso, r["country"]["value"], yr, round(g, 1), round(e, 1)))
    panel.sort(key=lambda p: (p[0], p[2]))
    return panel


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    return cov / math.sqrt(va * vb)


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def percentile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[int(idx)]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def analyse(panel):
    X = [math.log10(g) for _, _, _, g, _ in panel]
    Y = [math.log10(e) for _, _, _, _, e in panel]
    n = len(panel)
    mx, my = sum(X) / n, sum(Y) / n
    beta = sum((X[i] - mx) * (Y[i] - my) for i in range(n)) / sum((x - mx) ** 2 for x in X)
    alpha = my - beta * mx
    resid = [Y[i] - (alpha + beta * X[i]) for i in range(n)]
    sigma = math.sqrt(sum(e * e for e in resid) / (n - 2))
    r = pearson(X, Y)
    rho = pearson(ranks(X), ranks(Y))

    # within / between decomposition
    byc = defaultdict(list)
    for i, p in enumerate(panel):
        byc[p[0]].append(i)
    gbar = {c: sum(X[i] for i in idx) / len(idx) for c, idx in byc.items()}
    ebar = {c: sum(Y[i] for i in idx) / len(idx) for c, idx in byc.items()}
    rb = pearson([gbar[c] for c in byc], [ebar[c] for c in byc])
    wx, wy = [], []
    for c, idx in byc.items():
        if len(idx) >= 5:
            for i in idx:
                wx.append(X[i] - gbar[c])
                wy.append(Y[i] - ebar[c])
    rw = pearson(wx, wy)
    bw = sum(wx[i] * wy[i] for i in range(len(wx))) / sum(x * x for x in wx)

    # empty-corner support statement
    hi = [(p, e) for p, e in zip(panel, [10 ** y for y in Y]) if p[3] >= 30000]
    hi_min = min(hi, key=lambda t: t[0][4])[0]
    below = sum(1 for p, _ in hi if p[4] < LOW_KWH)

    return {
        "n": n, "beta": beta, "alpha": alpha, "sigma": sigma, "r": r, "rho": rho,
        "r2": r * r, "rb": rb, "rw": rw, "bw": bw, "ncountries": len(byc),
        "hi_n": len(hi), "hi_min": hi_min, "below": below,
        "X": X, "Y": Y,
    }


# --------------------------------------------------------------- density SVG
# sequential single-hue (blue) ramp, light -> dark, for log-count cells
RAMP = ["#e7edf4", "#c3d2e2", "#96afca", "#688cb0", "#436a92", "#274a6b"]


def build_density(panel, st):
    VW, VH = 760, 560
    PX0, PX1 = 92, 700          # plot area (leave room for density legend at right)
    PY0, PY1 = 30, 496
    X, Y = st["X"], st["Y"]
    XMIN, XMAX = min(X) - 0.06, max(X) + 0.06
    YMIN, YMAX = min(Y) - 0.10, max(Y) + 0.10
    NX, NY = 30, 26
    dx = (XMAX - XMIN) / NX
    dy = (YMAX - YMIN) / NY

    def sx(lg):   # lg = log10(gdp)
        return PX0 + (lg - XMIN) / (XMAX - XMIN) * (PX1 - PX0)

    def sy(le):   # le = log10(energy)
        return PY1 - (le - YMIN) / (YMAX - YMIN) * (PY1 - PY0)

    # 2-D histogram (track members per bin for hover tooltips)
    grid = defaultdict(int)
    members = defaultdict(list)
    for i in range(len(X)):
        ix = min(NX - 1, int((X[i] - XMIN) / dx))
        iy = min(NY - 1, int((Y[i] - YMIN) / dy))
        grid[(ix, iy)] += 1
        members[(ix, iy)].append((panel[i][1], panel[i][2]))  # (country, year)
    cmax = max(grid.values())

    def color(c):
        t = math.log(c) / math.log(cmax) if cmax > 1 else 1.0
        return RAMP[min(len(RAMP) - 1, int(t * (len(RAMP) - 1) + 1e-9))]

    s = [f'<svg viewBox="0 0 {VW} {VH}" role="img" aria-label="Density of {st["n"]} '
         f'country-year observations in the log GDP per capita by log energy use per '
         f'capita plane, {Y0YEAR}-{Y1YEAR}" xmlns="http://www.w3.org/2000/svg" class="scatter">']
    s.append('<style>'
             '.scatter text{font-family:Palatino,Georgia,serif;fill:#111}'
             '.ax{font-size:15px;fill:rgba(17,17,17,.55)}'
             '.axtitle{font-size:16px;font-style:italic;fill:rgba(17,17,17,.75)}'
             '.cell{stroke:#fffdf6;stroke-width:.5}'
             '.pt{fill:#1d2b3a;fill-opacity:.9}'
             '.lbl{font-size:12.5px;fill:#2d2d2d}'
             '.grid{stroke:rgba(17,17,17,.10);stroke-width:1}'
             '.ols{stroke:#8b2252;stroke-width:2;fill:none}'
             '.wall{stroke:#445c3c;stroke-width:2;fill:none;stroke-dasharray:6 3}'
             '.frame{stroke:rgba(17,17,17,.30);stroke-width:1;fill:none}'
             '.empty{fill:none;stroke:#a63d40;stroke-width:1.3;stroke-dasharray:5 4}'
             '.empty-lbl{font-size:13px;fill:#a63d40;font-style:italic}'
             '.leg{font-size:12px;fill:rgba(17,17,17,.6)}'
             '</style>')

    # cells
    for (ix, iy), c in sorted(grid.items()):
        x = sx(XMIN + ix * dx)
        y = sy(YMIN + (iy + 1) * dy)
        w = sx(XMIN + (ix + 1) * dx) - x
        h = sy(YMIN + iy * dy) - y
        bycountry = defaultdict(list)
        for cn, yr in members[(ix, iy)]:
            bycountry[cn].append(yr)
        tip = "\n".join(f"{cn}: {compact_years(yrs)}"
                        for cn, yrs in sorted(bycountry.items()))
        s.append(f'<rect class="cell" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
                 f'height="{h:.1f}" fill="{color(c)}" data-tip="{esc(tip)}"/>')

    # gridlines + ticks
    for g, lab in [(1000, '$1k'), (10000, '$10k'), (100000, '$100k')]:
        x = sx(math.log10(g))
        s.append(f'<line class="grid" x1="{x:.1f}" y1="{PY0}" x2="{x:.1f}" y2="{PY1}"/>')
        s.append(f'<text class="ax" x="{x:.1f}" y="{PY1 + 22}" text-anchor="middle">{lab}</text>')
    for e, lab in [(100, '100'), (1000, '1,000'), (10000, '10,000'), (100000, '100,000')]:
        y = sy(math.log10(e))
        s.append(f'<line class="grid" x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}"/>')
        s.append(f'<text class="ax" x="{PX0 - 10}" y="{y + 5:.1f}" text-anchor="end">{lab}</text>')

    # 5th-percentile lower envelope ("the wall"): per GDP bin
    wall = []
    for ix in range(NX):
        lo, hiv = XMIN + ix * dx, XMIN + (ix + 1) * dx
        ys = sorted(10 ** Y[i] for i in range(len(X)) if lo <= X[i] < hiv)
        if len(ys) >= 12:
            wall.append((sx((lo + hiv) / 2), sy(math.log10(percentile(ys, 0.05)))))
    if wall:
        s.append('<polyline class="wall" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in wall) + '"/>')

    # OLS line
    gx0, gx1 = 10 ** XMIN, 10 ** XMAX
    ly0 = st["alpha"] + st["beta"] * math.log10(gx0)
    ly1 = st["alpha"] + st["beta"] * math.log10(gx1)
    s.append(f'<line class="ols" x1="{sx(XMIN):.1f}" y1="{sy(ly0):.1f}" x2="{sx(XMAX):.1f}" y2="{sy(ly1):.1f}"/>')

    # empty-corner outline: G >= $30k and E < LOW_KWH
    ex0, ey0 = sx(math.log10(30000)), sy(math.log10(LOW_KWH))
    s.append(f'<rect class="empty" x="{ex0:.1f}" y="{ey0:.1f}" width="{PX1 - ex0:.1f}" height="{PY1 - ey0:.1f}"/>')
    cx = (ex0 + PX1) / 2
    s.append(f'<text class="empty-lbl" x="{cx:.1f}" y="{(ey0 + PY1) / 2 - 5:.1f}" text-anchor="middle">there are no rich,</text>')
    s.append(f'<text class="empty-lbl" x="{cx:.1f}" y="{(ey0 + PY1) / 2 + 12:.1f}" text-anchor="middle">low-energy countries</text>')

    # a few labelled extreme points (latest year). The petrostates (Turkmenistan,
    # Trinidad) sit high above the line: energy without the matching wealth.
    latest = {p[0]: p for p in panel if p[2] == Y1YEAR}
    marks = {'QAT': (-6, -7, 'end', 'Qatar'), 'NOR': (0, 16, 'middle', 'Norway'),
             'TTO': (0, -10, 'middle', 'Trinidad & Tobago'),
             'TKM': (-8, 4, 'end', 'Turkmenistan'),
             'MAC': (-8, 4, 'end', 'Macao'),
             'COD': (8, 4, 'start', 'DR Congo'),
             'SLE': (8, 4, 'start', 'Sierra Leone')}
    for iso, (dxp, dyp, anch, name) in marks.items():
        if iso not in latest:
            continue
        p = latest[iso]
        px, py = sx(math.log10(p[3])), sy(math.log10(p[4]))
        s.append(f'<circle class="pt" cx="{px:.1f}" cy="{py:.1f}" r="3.2"/>')
        s.append(f'<text class="lbl" x="{px + dxp:.1f}" y="{py + dyp:.1f}" text-anchor="{anch}">{esc(name)}</text>')

    # frame + axis titles
    s.append(f'<rect class="frame" x="{PX0}" y="{PY0}" width="{PX1 - PX0}" height="{PY1 - PY0}"/>')
    s.append(f'<text class="axtitle" x="{(PX0 + PX1) / 2:.1f}" y="{VH - 8}" text-anchor="middle">GDP per capita (constant 2015 US$, log scale)</text>')
    mid = (PY0 + PY1) / 2
    s.append(f'<text class="axtitle" x="22" y="{mid:.1f}" text-anchor="middle" transform="rotate(-90 22 {mid:.1f})">Primary energy use per capita (kWh/yr, log scale)</text>')

    # density legend (vertical ramp at right)
    lx, lw = 716, 14
    ly_top, lh = 120, 200
    seg = lh / len(RAMP)
    for k, col in enumerate(RAMP):
        s.append(f'<rect x="{lx}" y="{ly_top + (len(RAMP) - 1 - k) * seg:.1f}" width="{lw}" height="{seg:.1f}" fill="{col}"/>')
    s.append(f'<text class="leg" x="{lx + lw + 5}" y="{ly_top + 4}">{cmax}</text>')
    s.append(f'<text class="leg" x="{lx + lw + 5}" y="{ly_top + lh:.1f}">1</text>')
    s.append(f'<text class="leg" x="{lx + lw / 2:.0f}" y="{ly_top - 22}" text-anchor="middle">country-</text>')
    s.append(f'<text class="leg" x="{lx + lw / 2:.0f}" y="{ly_top - 22}" text-anchor="middle" dy="1.1em">years/cell</text>')
    s.append('</svg>')
    return '\n'.join(s)


def main():
    panel = load_panel()
    st = analyse(panel)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iso3", "country", "year", "gdp_per_capita_const2015usd", "primary_energy_per_capita_kwh"])
        for iso, name, yr, g, e in panel:
            w.writerow([iso, name, yr, g, e])

    OUT_SVG.write_text(build_density(panel, st))

    hm = st["hi_min"]
    print(f"wrote {OUT_CSV.name} ({st['n']:,} rows)")
    print(f"wrote {OUT_SVG.name}")
    print(f"n={st['n']}  beta={st['beta']:.3f}  r={st['r']:.3f}  rho={st['rho']:.3f}  "
          f"R2={st['r2']:.3f}  rB={st['rb']:.3f}  rW={st['rw']:.3f}  betaW={st['bw']:.3f}")
    print(f"sigma={st['sigma']:.3f} dex  ({st['ncountries']} countries)")
    print(f"empty corner: {st['below']} of {st['hi_n']} country-years with GDP >= $30k "
          f"below {LOW_KWH:,} kWh (lowest: {hm[4]:,.0f} kWh, {hm[1]} {hm[2]})")


if __name__ == "__main__":
    main()
