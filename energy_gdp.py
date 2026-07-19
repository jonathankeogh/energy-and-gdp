#!/usr/bin/env python3
"""Reproduce the joint distribution of GDP and energy across the 1980-2023 panel.

Reads the two saved raw datasets in ``data/``, merges them into a single
country-year panel, characterises the joint law of real GDP per capita and
primary energy use per capita, and writes:

    energy-gdp-panel-1980-2023.csv   the merged panel (one row per country-year)
    energy-gdp-chart.svg             the log-log density chart (self-contained)

and prints the headline statistics. Deterministic: same inputs -> same outputs.

    uv run python energy_gdp.py

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
import html
from pathlib import Path

import numpy as np
import pandas as pd

Y0YEAR, Y1YEAR = 1980, 2023

HERE = Path(__file__).resolve().parent
RAW = HERE / "data"

ENERGY_CSV = RAW / "owid-primary-energy-per-capita.csv"
GDP_JSON = RAW / "worldbank-gdp-per-capita-constant-2015-usd-1980-2024.json"
OUT_CSV = HERE / "energy-gdp-panel-1980-2023.csv"
OUT_SVG = HERE / "energy-gdp-chart.svg"

# Low-energy threshold for the "empty corner": no high-income (>= $30k, real
# 2015 US$) country-year falls below this many kWh of primary energy per capita.
LOW_KWH = 10_000
HIGH_GDP = 30_000

# World Bank aggregate/region codes to exclude so only sovereign entities remain.
AGG = {"AFE", "AFW", "ARB", "CEB", "CSS", "EAP", "EAR", "EAS", "ECA", "ECS",
       "EMU", "EUU", "FCS", "HIC", "HPC", "IBD", "IBT", "IDA", "IDB", "IDX",
       "LAC", "LCN", "LDC", "LIC", "LMC", "LMY", "LTE", "MEA", "MIC", "MNA",
       "NAC", "OED", "OSS", "PRE", "PSS", "PST", "SAS", "SSA", "SSF", "SST",
       "TEA", "TEC", "TLA", "TMN", "TSA", "TSS", "UMC", "WLD", "INX"}


# --------------------------------------------------------------- data & stats
def load_panel() -> pd.DataFrame:
    """Merge the two raw sources into one clean (iso3, country, year, gdp, energy) panel."""
    energy = pd.read_csv(ENERGY_CSV)
    energy = energy.loc[energy["code"].str.len() == 3, :].rename(
        columns={"code": "iso3", "primary_energy_consumption_per_capita__kwh": "energy"}
    )[["iso3", "year", "energy"]]

    records = json.loads(GDP_JSON.read_text())[1]
    gdp = pd.DataFrame(
        {
            "iso3": r["countryiso3code"],
            "country": r["country"]["value"],
            "year": int(r["date"]),
            "gdp": r["value"],
        }
        for r in records
    )
    gdp = gdp[~gdp["iso3"].isin(AGG) & (gdp["iso3"].str.len() == 3)]
    gdp = gdp[gdp["year"].between(Y0YEAR, Y1YEAR)]

    df = gdp.merge(energy, on=["iso3", "year"], how="inner").dropna(subset=["gdp", "energy"])
    df = df[(df["gdp"] > 0) & (df["energy"] > 0)].copy()
    df["gdp"] = df["gdp"].round(1)
    df["energy"] = df["energy"].round(1)
    return df.sort_values(["iso3", "year"]).reset_index(drop=True)[
        ["iso3", "country", "year", "gdp", "energy"]
    ]


def analyse(df: pd.DataFrame) -> dict:
    x = np.log10(df["gdp"].to_numpy())
    y = np.log10(df["energy"].to_numpy())
    n = len(df)

    # pooled OLS in log-log space: beta is the elasticity of energy w.r.t. income
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    sigma = float(np.sqrt(np.sum(resid**2) / (n - 2)))
    r = float(np.corrcoef(x, y)[0, 1])
    # Spearman rho = Pearson correlation of the (average-tie) ranks
    rho = float(pd.Series(x).rank().corr(pd.Series(y).rank()))

    # between vs. within decomposition
    logs = df.assign(lx=x, ly=y)
    grp = logs.groupby("iso3")
    means = grp[["lx", "ly"]].mean()
    rb = float(means["lx"].corr(means["ly"]))

    dev = logs.assign(
        wx=logs["lx"] - grp["lx"].transform("mean"),
        wy=logs["ly"] - grp["ly"].transform("mean"),
        n_obs=grp["year"].transform("size"),
    )
    dev = dev[dev["n_obs"] >= 5]
    wx, wy = dev["wx"].to_numpy(), dev["wy"].to_numpy()
    rw = float(np.corrcoef(wx, wy)[0, 1])
    bw = float(np.polyfit(wx, wy, 1)[0])

    # empty-corner support statement
    hi = df[df["gdp"] >= HIGH_GDP]
    hi_min = hi.loc[hi["energy"].idxmin()]
    below = int((hi["energy"] < LOW_KWH).sum())

    return {
        "n": n, "beta": float(beta), "alpha": float(alpha), "sigma": sigma,
        "r": r, "rho": rho, "r2": r * r, "rb": rb, "rw": rw, "bw": bw,
        "ncountries": int(df["iso3"].nunique()),
        "hi_n": len(hi), "hi_min": hi_min, "below": below,
        "x": x, "y": y,
    }


# --------------------------------------------------------------- density SVG
# sequential single-hue (blue) ramp, light -> dark, for log-count cells
RAMP = ["#e7edf4", "#c3d2e2", "#96afca", "#688cb0", "#436a92", "#274a6b"]


def esc(s):
    return html.escape(str(s))


def compact_years(years):
    """[2001, 2002, 2003, 2005] -> '2001-2003, 2005'"""
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


def build_density(df: pd.DataFrame, st: dict) -> str:
    VW, VH = 760, 560
    PX0, PX1 = 92, 700          # plot area (leave room for density legend at right)
    PY0, PY1 = 30, 496
    x, y = st["x"], st["y"]
    XMIN, XMAX = x.min() - 0.06, x.max() + 0.06
    YMIN, YMAX = y.min() - 0.10, y.max() + 0.10
    NX, NY = 30, 26
    dx = (XMAX - XMIN) / NX
    dy = (YMAX - YMIN) / NY

    def sx(lg):
        return PX0 + (lg - XMIN) / (XMAX - XMIN) * (PX1 - PX0)

    def sy(le):
        return PY1 - (le - YMIN) / (YMAX - YMIN) * (PY1 - PY0)

    # 2-D histogram, tracking members per bin for hover tooltips
    ix = np.minimum(NX - 1, ((x - XMIN) / dx).astype(int))
    iy = np.minimum(NY - 1, ((y - YMIN) / dy).astype(int))
    binned = df.assign(ix=ix, iy=iy)
    counts = binned.groupby(["ix", "iy"]).size()
    cmax = int(counts.max())

    def color(c):
        t = np.log(c) / np.log(cmax) if cmax > 1 else 1.0
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
    for (bx, by), members in binned.groupby(["ix", "iy"]):
        c = len(members)
        x0 = sx(XMIN + bx * dx)
        y0 = sy(YMIN + (by + 1) * dy)
        w = sx(XMIN + (bx + 1) * dx) - x0
        h = sy(YMIN + by * dy) - y0
        tip = "\n".join(
            f"{country}: {compact_years(list(g['year']))}"
            for country, g in members.groupby("country")
        )
        s.append(f'<rect class="cell" x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" '
                 f'height="{h:.1f}" fill="{color(c)}" data-tip="{esc(tip)}"/>')

    # gridlines + ticks
    for g, lab in [(1000, '$1k'), (10000, '$10k'), (100000, '$100k')]:
        gx = sx(np.log10(g))
        s.append(f'<line class="grid" x1="{gx:.1f}" y1="{PY0}" x2="{gx:.1f}" y2="{PY1}"/>')
        s.append(f'<text class="ax" x="{gx:.1f}" y="{PY1 + 22}" text-anchor="middle">{lab}</text>')
    for e, lab in [(100, '100'), (1000, '1,000'), (10000, '10,000'), (100000, '100,000')]:
        gy = sy(np.log10(e))
        s.append(f'<line class="grid" x1="{PX0}" y1="{gy:.1f}" x2="{PX1}" y2="{gy:.1f}"/>')
        s.append(f'<text class="ax" x="{PX0 - 10}" y="{gy + 5:.1f}" text-anchor="end">{lab}</text>')

    # 5th-percentile lower envelope ("the wall"): per GDP bin
    wall = []
    for bx in range(NX):
        lo, hiv = XMIN + bx * dx, XMIN + (bx + 1) * dx
        ys = df["energy"].to_numpy()[(x >= lo) & (x < hiv)]
        if len(ys) >= 12:
            wall.append((sx((lo + hiv) / 2), sy(np.log10(np.percentile(ys, 5)))))
    if wall:
        s.append('<polyline class="wall" points="' + ' '.join(f'{a:.1f},{b:.1f}' for a, b in wall) + '"/>')

    # pooled OLS line
    ly0 = st["alpha"] + st["beta"] * XMIN
    ly1 = st["alpha"] + st["beta"] * XMAX
    s.append(f'<line class="ols" x1="{sx(XMIN):.1f}" y1="{sy(ly0):.1f}" x2="{sx(XMAX):.1f}" y2="{sy(ly1):.1f}"/>')

    # empty-corner outline: GDP >= HIGH_GDP and energy < LOW_KWH
    ex0, ey0 = sx(np.log10(HIGH_GDP)), sy(np.log10(LOW_KWH))
    s.append(f'<rect class="empty" x="{ex0:.1f}" y="{ey0:.1f}" width="{PX1 - ex0:.1f}" height="{PY1 - ey0:.1f}"/>')
    cx = (ex0 + PX1) / 2
    s.append(f'<text class="empty-lbl" x="{cx:.1f}" y="{(ey0 + PY1) / 2 - 5:.1f}" text-anchor="middle">there are no rich,</text>')
    s.append(f'<text class="empty-lbl" x="{cx:.1f}" y="{(ey0 + PY1) / 2 + 12:.1f}" text-anchor="middle">low-energy countries</text>')

    # a few labelled extreme points (latest year)
    latest = df[df["year"] == Y1YEAR].set_index("iso3")
    marks = {'QAT': (-6, -7, 'end', 'Qatar'), 'NOR': (0, 16, 'middle', 'Norway'),
             'TTO': (0, -10, 'middle', 'Trinidad & Tobago'),
             'TKM': (-8, 4, 'end', 'Turkmenistan'),
             'MAC': (-8, 4, 'end', 'Macao'),
             'COD': (8, 4, 'start', 'DR Congo'),
             'SLE': (8, 4, 'start', 'Sierra Leone')}
    for iso, (dxp, dyp, anch, name) in marks.items():
        if iso not in latest.index:
            continue
        row = latest.loc[iso]
        px, py = sx(np.log10(row["gdp"])), sy(np.log10(row["energy"]))
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
    df = load_panel()
    st = analyse(df)

    df.rename(
        columns={"gdp": "gdp_per_capita_const2015usd", "energy": "primary_energy_per_capita_kwh"}
    ).to_csv(OUT_CSV, index=False)
    OUT_SVG.write_text(build_density(df, st))

    hm = st["hi_min"]
    print(f"wrote {OUT_CSV.name} ({st['n']:,} rows)")
    print(f"wrote {OUT_SVG.name}")
    print(f"n={st['n']}  beta={st['beta']:.3f}  r={st['r']:.3f}  rho={st['rho']:.3f}  "
          f"R2={st['r2']:.3f}  rB={st['rb']:.3f}  rW={st['rw']:.3f}  betaW={st['bw']:.3f}")
    print(f"sigma={st['sigma']:.3f} dex  ({st['ncountries']} countries)")
    print(f"empty corner: {st['below']} of {st['hi_n']} country-years with GDP >= ${HIGH_GDP:,} "
          f"below {LOW_KWH:,} kWh (lowest: {hm['energy']:,.0f} kWh, {hm['country']} {hm['year']})")


if __name__ == "__main__":
    main()
