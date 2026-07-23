"""Channel bookkeeping and the stack of probability panels.

The produced flavour can be nu_e, nu_mu or nu_tau; all three final states are
always plotted (survival first, then the two appearance channels). Colours
follow the DETECTED flavour, so gold always means muon-flavour, blue
electron-flavour and rose tau-flavour, whichever flavour was produced.
"""
import matplotlib.pyplot as plt

from jaxnu import Flavor

GOLD_D, GOLD_L = "#9C6404", "#E0A93F"     # numu / numubar
BLUE_D, BLUE_L = "#1B5A85", "#6FA8CE"     # nue  / nuebar
ROSE_D, ROSE_L = "#A33A5B", "#DE8CA8"     # nutau / nutaubar
INK, MUTED = "#2B2F36", "#6B7280"

CHAN_COL = {Flavor.MU: (GOLD_D, GOLD_L),
            Flavor.E: (BLUE_D, BLUE_L),
            Flavor.TAU: (ROSE_D, ROSE_L)}
TEX = {Flavor.MU: r"\nu_\mu", Flavor.E: r"\nu_e", Flavor.TAU: r"\nu_\tau"}
BAR = {Flavor.MU: r"\bar\nu_\mu", Flavor.E: r"\bar\nu_e", Flavor.TAU: r"\bar\nu_\tau"}
ALL_FLAVOURS = (Flavor.E, Flavor.MU, Flavor.TAU)


def channel_list(init):
    """Final-state flavours to plot: survival first, then the appearances."""
    return [init] + [f for f in ALL_FLAVOURS if f != init]


def style_axes(*axes):
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(color="#DDE1E7", lw=0.8, alpha=1.0)
        for sp in ax.spines.values():
            sp.set_color("#C7CCD4")
        ax.tick_params(colors=MUTED, labelcolor=INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_color(INK)


def apply_mode(axes, lines, init, fs=13):
    """Re-label/re-colour existing panels for a new produced flavour.

    Only the artists change, so the axes never need rebuilding (which keeps the
    blitting set-up intact). Returns the channel list."""
    chans = channel_list(init)
    for ax, (ln, lb), fout in zip(axes, lines, chans):
        cd, cl = CHAN_COL[fout]
        ln.set_color(cd); ln.set_label(fr"${TEX[init]} \to {TEX[fout]}$")
        lb.set_color(cl); lb.set_label(fr"${BAR[init]} \to {BAR[fout]}$")
        ax.set_ylabel(fr"$P({TEX[init]} \to {TEX[fout]})$", fontsize=fs)
        survival = (fout == init)
        ax.set_ylim(0, 1.02 if survival else 0.09)
        ax.legend(frameon=False, ncol=2, fontsize=fs - 3,
                  loc="lower right" if survival else "upper right")
    return chans


def build_panels(fig, init, top=0.945, bot=0.685, gap=0.014,
                 left=0.12, width=0.83, fs=13):
    """One panel per final-state flavour. lines[i] = (neutrino, antineutrino)."""
    n = len(ALL_FLAVOURS)
    h = (top - bot - gap * (n - 1)) / n
    axes, lines = [], []
    for i in range(n):
        rect = [left, top - h - i * (h + gap), width, h]
        ax = fig.add_axes(rect, sharex=axes[0] if axes else None)
        (ln,) = ax.plot([], [])
        (lb,) = ax.plot([], [], ls="--")
        axes.append(ax)
        lines.append((ln, lb))
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
    axes[-1].set_xlabel(r"$E_\nu$  (GeV)")
    style_axes(*axes)
    apply_mode(axes, lines, init, fs=fs)
    return axes, lines
