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
SHORT = {Flavor.MU: r"\mu", Flavor.E: "e", Flavor.TAU: r"\tau"}


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


def fit_labels(fig, axes, init):
    """Size the panel labels to the current window.

    The y-label is rotated, so its *height* is the text length: at the default
    window ``P(nu_mu -> nu_tau)`` is already 0.84 in inside a 1.0 in panel, and
    shrinking the window makes it spill into the neighbouring panel. Scale the
    font with the panel height and drop to a compact ``P_ab`` form once the
    panels get too short for the full arrow notation.
    """
    ph_in = fig.get_figheight() * axes[0].get_position().height
    fs = max(6.0, min(14.0, ph_in * 13.0))
    compact = ph_in < 0.95
    for ax, fout in zip(axes, channel_list(init)):
        sub = f"{SHORT[init]}\\,{SHORT[fout]}"     # thin space: \mu e, not \mue
        ax.set_ylabel(
            f"$P_{{{sub}}}$" if compact
            else fr"$P({TEX[init]} \to {TEX[fout]})$", fontsize=fs)
        ax.tick_params(labelsize=max(6.0, fs - 2))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                t.set_fontsize(max(6.0, fs - 3))
    axes[-1].xaxis.label.set_size(max(7.0, fs + 1))
    return fs


def apply_mode(axes, lines, init, fs=13):
    """Re-label/re-colour existing panels for a new produced flavour.

    Only the artists change, so the axes never need rebuilding (which keeps the
    blitting set-up intact). Returns the channel list."""
    chans = channel_list(init)
    for ax, (ln, lb), fout in zip(axes, lines, chans):
        cd, cl = CHAN_COL[fout]
        ln.set_color(cd); ln.set_label(fr"${TEX[init]} \to {TEX[fout]}$")
        lb.set_color(cl); lb.set_label(fr"${BAR[init]} \to {BAR[fout]}$")
        survival = (fout == init)
        ax.set_ylim(0, 1.02 if survival else 0.09)
        ax.legend(frameon=False, ncol=2, fontsize=fs - 3,
                  loc="lower right" if survival else "upper right")
    fit_labels(axes[0].get_figure(), axes, init)
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
