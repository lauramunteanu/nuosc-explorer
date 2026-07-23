"""Talk-quality matplotlib style + helpers for jaxnu plots.

Import at the top of any plotting script:

    from nuosc_style import use_talk_style, PASTEL, save

`use_talk_style()` sets large, slide-friendly fonts and a soft, colorblind-safe
pastel palette. `save(fig, "name")` writes a vector PDF (LaTeX/Keynote) and a
300-dpi PNG into ./figures.
"""
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

FIGDIR = Path(os.environ.get("NUOSC_FIGDIR", "figures")).resolve()

# Soft, colorblind-safe pastel palette (muted Okabe-Ito style ordering:
# blue / amber / green / mauve / coral / teal — max hue separation under CVD).
PASTEL = [
    "#5A9BD4",   # soft blue
    "#E8A85C",   # soft amber
    "#5FB894",   # soft green
    "#E48AB8",   # soft pink
    "#E27D6A",   # soft coral
    "#6EC0C9",   # soft teal
    "#C77CB0",   # soft mauve
]

# Perceptually-uniform, colorblind-safe sequential map for oscillograms.
OSCILLOGRAM_CMAP = "cividis"


def use_talk_style():
    """rcParams for slides: large fonts, pastel cycle, clean vector output."""
    mpl.rcParams.update({
        # Large fonts, readable from the back of a lecture hall.
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        # Lines / marks (a touch thick so pastels stay vivid on a projector).
        "lines.linewidth": 3.0,
        "lines.markersize": 8,
        "axes.prop_cycle": cycler(color=PASTEL),
        # Axes: recessive grid, no top/right spines.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "axes.axisbelow": True,
        "axes.linewidth": 1.1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Output.
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,      # searchable/editable text in the PDF
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save(fig, name):
    """Save `fig` as both PDF (vector) and PNG (raster) into ./figures."""
    FIGDIR.mkdir(exist_ok=True)
    pdf, png = FIGDIR / f"{name}.pdf", FIGDIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    print(f"saved {pdf}")
    print(f"saved {png}")
    return pdf, png
