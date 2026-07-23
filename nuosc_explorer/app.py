"""Interactive neutrino-oscillation GUI (with matter NSI).

Vary every oscillation parameter (and the matter-NSI epsilons) with sliders or
typed entry boxes, load published best-fit presets, and export the current view
as a PNG or sweep one parameter into a GIF. An optional Analysis window (opened
on demand) exploits jaxnu's autodiff.

Run inside the `jaxnu` conda env (opens a native window):
    conda activate jaxnu
    python gui.py

Controls
--------
  left card  : sin^2 th12/th13/th23, dm21, dm32, delta_CP, baseline L, E range
               each slider has a typed entry box and its +/- uncertainty
  right card : NSI eps_ee, eps_emu, eps_etau, eps_mutau (real, =0 -> standard)
  Presets    : load published central values + uncertainties
  Ordering   : Normal / Inverted
  Sweep      : which parameter the "Save GIF" button animates
  Save PNG / Save GIF / Analysis
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation, PillowWriter

from jaxnu import probability_constant, OscParams, NSI, Flavor

from .style import use_talk_style, FIGDIR
from . import panels
from .panels import GOLD_D, GOLD_L, BLUE_D, BLUE_L, ROSE_D, ROSE_L

use_talk_style()


def _patch_mpl_textbox_resize():
    """Work around a matplotlib bug (<= 3.11).

    ``TextBox.__init__`` connects ``_resize`` to ``'resize_event'``, but
    ``_resize`` is wrapped in ``_call_with_reparented_event``, which reads
    ``event.inaxes`` -- an attribute ``ResizeEvent`` does not have. Resizing a
    window containing a TextBox therefore raises AttributeError. The real body
    only calls ``stop_typing()`` and ignores the event, so use the undecorated
    original (``functools.wraps`` keeps it on ``__wrapped__``). A no-op if
    upstream drops the decorator.
    """
    original = getattr(TextBox._resize, "__wrapped__", None)
    if original is not None:
        TextBox._resize = original


_patch_mpl_textbox_resize()

NE = 500                      # fixed length -> changing the E range never recompiles
DENSITY, YE, FPS = 2.6, 0.5, 20

# ---- UI palette -----------------------------------------------------------
UI_BG    = "#F4F5F7"
GROUP_BG = "#E9ECF1"
TRACK    = "#D6DAE1"
ACCENT   = "#4C7FB8"
INK      = "#2B2F36"
MUTED    = "#6B7280"

# ===========================================================================
# Published parameter sets: value and (+sigma, -sigma), in slider units
# (angles as sin^2, dm21 in 1e-5 eV^2, dm32 in 1e-3 eV^2, delta_CP in rad).
#
# !! APPROXIMATE - recalled from the literature, normal ordering, and rounded.
# !! Check against the exact reference you intend to cite before using in a
# !! talk; they are all in this one dict, so they are trivial to correct.
# ===========================================================================
PRESETS = {
    "T2K": {   # T2K best fit (solar sector taken from the global fits)
        "s12": (0.307, (0.013, 0.013)),   "s13": (0.0216, (0.0007, 0.0006)),
        "s23": (0.539, (0.031, 0.069)),   "dm21": (7.42, (0.21, 0.20)),
        "dm32": (2.502, (0.052, 0.044)),  "dcp": (-2.04, (0.81, 0.56))},
    "NOvA": {  # NOvA best fit (theta13 from the reactor constraint)
        "s12": (0.307, (0.013, 0.013)),   "s13": (0.0216, (0.0007, 0.0007)),
        "s23": (0.570, (0.030, 0.040)),   "dm21": (7.42, (0.21, 0.20)),
        "dm32": (2.410, (0.070, 0.070)),  "dcp": (2.58, (0.90, 0.90))},
    "PDG": {
        "s12": (0.307, (0.013, 0.013)),   "s13": (0.0220, (0.0007, 0.0007)),
        "s23": (0.546, (0.021, 0.021)),   "dm21": (7.53, (0.18, 0.18)),
        "dm32": (2.455, (0.028, 0.028)),  "dcp": (-2.54, (1.10, 0.79))},
    "NuFIT": {
        "s12": (0.303, (0.012, 0.011)),   "s13": (0.02203, (0.00056, 0.00058)),
        "s23": (0.451, (0.019, 0.016)),   "dm21": (7.41, (0.21, 0.20)),
        "dm32": (2.433, (0.026, 0.027)),  "dcp": (-2.23, (0.63, 0.45))},
}


def _nice_top(vmax, cap=1.02):
    """Round a maximum up to a tidy axis limit (1 / 1.5 / 2 / 3 / 5 / 7.5 x 10^n)."""
    if not np.isfinite(vmax) or vmax <= 0:
        return 0.01
    exp = np.floor(np.log10(vmax))
    frac = vmax / 10.0 ** exp
    for m in (1, 1.5, 2, 3, 5, 7.5):
        if frac <= m:
            return min(cap, float(m * 10.0 ** exp))
    return min(cap, float(10.0 ** (exp + 1)))


def _finite(*arrays):
    """Concatenate and drop non-finite samples.

    P is undefined at E = 0 (the oscillation phase divides by E) and jaxnu
    returns NaN there. NaN poisons min()/max() and every comparison against it
    is False, which silently defeats both the autoscaling and the clip check.
    """
    ys = np.concatenate([np.asarray(a).ravel() for a in arrays])
    return ys[np.isfinite(ys)]


def _nice_span(lo, hi):
    """Tidy (bottom, top) for a survival panel.

    Keep the conventional 0 floor when the dip actually gets near zero,
    otherwise zoom to the band the curves occupy (rounded to 0.05) so a shallow
    oscillation does not sit squashed against the top of an empty axis."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return 0.0, 1.02
    if lo < 0.15:
        return 0.0, 1.02
    pad = 0.08 * (hi - lo) + 1e-3
    bot = max(0.0, np.floor((lo - pad) * 20) / 20)
    top = min(1.02, np.ceil((hi + pad) * 20) / 20)
    return bot, (top if top > bot else bot + 0.05)


def _unc_str(u):
    if u is None:
        return ""
    p, m = u
    return f"±{p:g}" if abs(p - m) < 1e-12 else f"+{p:g} −{m:g}"


_CHAN = {}


def _chan(fin, fout, anti):
    """Cached jitted P(fin -> fout); params/L/nsi/E traced -> no recompile."""
    key = (fin, fout, anti)
    if key not in _CHAN:
        _CHAN[key] = jax.jit(lambda p, L, nsi, e: probability_constant(
            p, e, L, density=DENSITY, ye=YE, anti=anti, nsi=nsi,
            flavor_in=fin, flavor_out=fout))
    return _CHAN[key]


def _theta(sin2):
    return float(np.arcsin(np.sqrt(sin2)))


def build_params(s12, s13, s23, dm21, dm32, dcp, ordering):
    mag = dm32 + dm21
    dm31 = mag if ordering == "Normal" else -mag
    return OscParams(theta12=_theta(s12), theta13=_theta(s13), theta23=_theta(s23),
                     deltacp=float(dcp), dm21=float(dm21), dm31=float(dm31))


def build_nsi(ee, emu, etau, mutau):
    return NSI(eps_ee=float(ee), eps_emu=float(emu),
               eps_etau=float(etau), eps_mutau=float(mutau))


def compute(params, L, nsi, e, init):
    """(nu, nubar) arrays for each channel of `init`, survival first."""
    L = float(L)
    out = []
    for fout in panels.channel_list(init):
        for anti in (False, True):
            out.append(np.asarray(
                _chan(init, fout, anti)(params, L, nsi, e).block_until_ready()))
    return out


# =====================================================================
# Differentiable analyses (this is where jaxnu's autodiff is exploited).
# =====================================================================
def _osc_from(base, s13=None, s23=None, dm32=None, dcp=None):
    """Rebuild OscParams, replacing selected fields with (possibly traced)
    values in slider space (sin^2 for angles). Differentiable."""
    th13 = jnp.arcsin(jnp.sqrt(s13)) if s13 is not None else base.theta13
    th23 = jnp.arcsin(jnp.sqrt(s23)) if s23 is not None else base.theta23
    dcpv = dcp if dcp is not None else base.deltacp
    dm31 = (jnp.sign(base.dm31) * (dm32 + base.dm21)) if dm32 is not None else base.dm31
    return dataclasses.replace(base, theta13=th13, theta23=th23,
                               deltacp=dcpv, dm31=dm31)


def _P(params, L, nsi, e, anti, fin, fout):
    return probability_constant(params, e, L, density=DENSITY, ye=YE,
                                anti=anti, nsi=nsi, flavor_in=fin, flavor_out=fout)


_JAC, _BJAC = {}, {}


def sens_jac(pname, anti, fin, fout):
    """d P(E) / d <one parameter>; jitted f(x, base, L, nsi, e)."""
    key = (pname, anti, fin, fout)
    if key not in _JAC:
        def f(x, base, L, nsi, e):
            return _P(_osc_from(base, **{pname: x}), L, nsi, e, anti, fin, fout)
        _JAC[key] = jax.jit(jax.jacfwd(f))
    return _JAC[key]


def band_jac(anti, fin, fout):
    """d P(E) / d [s13, s23, dm32, dcp]; jitted, shape (nE, 4)."""
    key = (anti, fin, fout)
    if key not in _BJAC:
        def g(vec, base, L, nsi, e):
            p = _osc_from(base, s13=vec[0], s23=vec[1], dm32=vec[2], dcp=vec[3])
            return _P(p, L, nsi, e, anti, fin, fout)
        _BJAC[key] = jax.jit(jax.jacfwd(g))
    return _BJAC[key]


# label, min, max, display multiplier  (physical units)
CONTOUR_RANGE = {
    "s23":  (r"$\sin^2\theta_{23}$", 0.35, 0.65, 1.0),
    "dm32": (r"$\Delta m^2_{32}\ [10^{-3}\,\mathrm{eV}^2]$", 2.20e-3, 2.80e-3, 1e3),
    "s13":  (r"$\sin^2\theta_{13}$", 0.015, 0.030, 1.0),
    "dcp":  (r"$\delta_{CP}$ [rad]", -np.pi, np.pi, 1.0),
}


def contour_data(base, L, nsi, e, xk, yk, x0, y0, na=41, nb=41,
                 init=Flavor.MU, other=Flavor.E):
    """Asimov chi^2 over any two parameters.

    ILLUSTRATIVE ONLY. The "data" are the probabilities at the current slider
    values (x0, y0) with invented per-bin errors; there are no event rates,
    flux, cross-sections or systematics, and nothing is marginalised - every
    other parameter is held fixed. The best fit therefore sits exactly on the
    truth by construction; the contours show the resulting uncertainty region.
    The gradient fit (jax.grad + Adam from an offset start) demonstrates that
    the autodiff optimiser converges - it is a machinery check, not a result.
    """
    idx = jnp.arange(0, NE, max(1, NE // 25))          # ~25 "measurements"
    erra, errd = 0.010, 0.050
    truth = _osc_from(base, **{xk: x0, yk: y0})
    d_app = _P(truth, L, nsi, e, False, init, other)[idx]
    d_dis = _P(truth, L, nsi, e, False, init, init)[idx]

    def chi2(xv, yv):
        p = _osc_from(base, **{xk: xv, yk: yv})
        pa = _P(p, L, nsi, e, False, init, other)[idx]
        pd = _P(p, L, nsi, e, False, init, init)[idx]
        return jnp.sum((pa - d_app) ** 2) / erra ** 2 \
            + jnp.sum((pd - d_dis) ** 2) / errd ** 2

    _, xlo, xhi, _ = CONTOUR_RANGE[xk]
    _, ylo, yhi, _ = CONTOUR_RANGE[yk]
    xs = jnp.linspace(xlo, xhi, na)
    ys = jnp.linspace(ylo, yhi, nb)
    grid = jax.jit(lambda: jax.vmap(lambda b: jax.vmap(lambda a: chi2(a, b))(xs))(ys))()
    grid = np.array(grid); grid -= grid.min()

    # gradient fit in normalised coords so Adam is well scaled
    xc, xr = 0.5 * (xlo + xhi), 0.5 * (xhi - xlo)
    yc, yr = 0.5 * (ylo + yhi), 0.5 * (yhi - ylo)

    def loss(u):
        return chi2(xc + xr * u[0], yc + yr * u[1])

    gL = jax.jit(jax.grad(loss))
    u = jnp.array([(x0 - xc) / xr + 0.35, (y0 - yc) / yr - 0.35])   # offset start
    m = v = jnp.zeros(2)
    for t in range(1, 251):
        g = gL(u)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        u = u - 0.02 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    fit = (float(xc + xr * u[0]), float(yc + yr * u[1]))
    return np.asarray(xs), np.asarray(ys), grid, (float(x0), float(y0)), fit


# ---------------------------------------------------------------- widgets
def _style_radio(ax, labels, fontsize=13, active=0):
    ax.set_facecolor(GROUP_BG)
    for sp in ax.spines.values():
        sp.set_visible(False)
    r = RadioButtons(ax, labels, active=active, activecolor=ACCENT)
    for t in r.labels:
        t.set_fontsize(fontsize); t.set_color(INK)
    return r


def _style_entry(ax, initial):
    tb = TextBox(ax, "", initial=initial, color="white", hovercolor="#EAF0F7",
                 textalignment="center")
    tb.text_disp.set_fontsize(12); tb.text_disp.set_color(INK)
    for sp in ax.spines.values():
        sp.set_color("#C7CCD4")
    return tb


class OscGUI:
    # (label, key, vmin, vmax, init, display-scale, fmt)
    STD = [
        (r"$\sin^2\theta_{12}$", "s12", 0.25, 0.36, 0.307, 1, "%.3f"),
        (r"$\sin^2\theta_{13}$", "s13", 0.015, 0.030, 0.0220, 1, "%.4f"),
        (r"$\sin^2\theta_{23}$", "s23", 0.35, 0.65, 0.546, 1, "%.3f"),
        (r"$\Delta m^2_{21}\,[10^{-5}]$", "dm21", 6.5, 8.0, 7.53, 1e-5, "%.2f"),
        (r"$\Delta m^2_{32}\,[10^{-3}]$", "dm32", 2.20, 2.80, 2.455, 1e-3, "%.3f"),
        (r"$\delta_{CP}$ [rad]", "dcp", -np.pi, np.pi, -2.54, 1, "%.2f"),
        (r"$L$ [km]", "L", 100, 2000, 295, 1, "%.0f"),
    ]
    NSI = [
        (r"$\epsilon_{ee}$", "eps_ee", -0.3, 0.3, 0.0, 1, "%.2f"),
        (r"$\epsilon_{e\mu}$", "eps_emu", -0.3, 0.3, 0.0, 1, "%.2f"),
        (r"$\epsilon_{e\tau}$", "eps_etau", -0.3, 0.3, 0.0, 1, "%.2f"),
        (r"$\epsilon_{\mu\tau}$", "eps_mutau", -0.3, 0.3, 0.0, 1, "%.2f"),
    ]

    def __init__(self):
        self._listeners, self._syncing = [], False
        self.analysis = None
        self.emin, self.emax = 0.05, 3.0
        self.init_flav = Flavor.MU
        self.fig = plt.figure(figsize=(13.5, 13.0), dpi=100, facecolor=UI_BG)

        self.bg = self.fig.add_axes([0, 0, 1, 1], zorder=-1)
        self.bg.axis("off"); self.bg.patch.set_visible(False)
        self.bg.set_xlim(0, 1); self.bg.set_ylim(0, 1)
        for x0, y0, x1, y1 in [(0.085, 0.145, 0.63, 0.585),   # oscillation params
                               (0.695, 0.345, 0.985, 0.585),  # NSI
                               (0.695, 0.215, 0.985, 0.335),  # ordering
                               (0.695, 0.010, 0.985, 0.205),  # sweep
                               (0.085, 0.078, 0.63, 0.138),   # presets
                               (0.085, 0.005, 0.63, 0.072)]:  # actions
            self.bg.add_patch(FancyBboxPatch(
                (x0, y0), x1 - x0, y1 - y0,
                boxstyle="round,pad=0.004,rounding_size=0.012",
                linewidth=0, facecolor=GROUP_BG, zorder=0))

        self.axes, self.lines = panels.build_panels(self.fig, self.init_flav)
        self.title = self.axes[0].set_title("")
        self.header = self.fig.text(
            0.085, 0.988, "jaxnu  ·  neutrino oscillation explorer",
            fontsize=17, weight="bold", color=INK, va="top")

        # ---- parameter sliders + entry boxes + uncertainties
        self.sliders, self.entries, self.unc, self.fmt = {}, {}, {}, {}
        self._header(0.10, 0.607, "Oscillation parameters")
        self._add_sliders(self.STD, x=0.26, w=0.175, y0=0.570,
                          ex=0.447, ew=0.058, unc_x=0.515)
        self._header(0.71, 0.607, r"Matter NSI  $\epsilon_{\alpha\beta}$")
        self._add_sliders(self.NSI, x=0.80, w=0.085, y0=0.570,
                          ex=0.895, ew=0.055, unc_x=None)

        # ---- energy range row (bottom of the parameter card)
        self.fig.text(0.255, 0.247, r"$E_\nu$ range [GeV]", ha="right", va="bottom",
                      fontsize=13, color=INK)
        self.e_lo = _style_entry(self.fig.add_axes([0.26, 0.240, 0.058, 0.026]),
                                 f"{self.emin:g}")
        self.e_hi = _style_entry(self.fig.add_axes([0.330, 0.240, 0.058, 0.026]),
                                 f"{self.emax:g}")
        self.e_lo.on_submit(lambda t: self._on_energy())
        self.e_hi.on_submit(lambda t: self._on_energy())

        self._header(0.705, 0.360, r"Initial $\nu$")
        self.radio_flav = _style_radio(
            self.fig.add_axes([0.710, 0.262, 0.115, 0.090]),
            (r"$\nu_e$", r"$\nu_\mu$", r"$\nu_\tau$"), active=1)
        self.radio_flav.on_clicked(self._set_flavour)

        self._header(0.850, 0.360, "Ordering")
        self.radio_mo = _style_radio(self.fig.add_axes([0.855, 0.276, 0.125, 0.070]),
                                     ("Normal", "Inverted"))
        self.radio_mo.on_clicked(lambda _l: self._refresh())

        self._header(0.705, 0.196, "Sweep (for GIF)")
        self.radio_sweep = _style_radio(
            self.fig.add_axes([0.712, 0.020, 0.20, 0.165]),
            (r"$\delta_{CP}$", r"$\sin^2\theta_{23}$", r"$\Delta m^2_{32}$",
             r"$\epsilon_{e\mu}$", r"$\epsilon_{e\tau}$", "Ordering"))

        # ---- presets
        self.fig.text(0.098, 0.168, "Presets", fontsize=13, weight="bold", color=INK)
        self.preset_btns = []
        for i, name in enumerate(PRESETS):
            b = self._button([0.175 + i * 0.113, 0.115, 0.103, 0.040], name)
            b.on_clicked(lambda _e, n=name: self._apply_preset(n))
            self.preset_btns.append(b)

        # ---- actions
        self.btn_png = self._button([0.098, 0.048, 0.125, 0.040], "Save PNG")
        self.btn_png.on_clicked(self.save_png)
        self.btn_gif = self._button([0.238, 0.048, 0.125, 0.040], "Save GIF")
        self.btn_gif.on_clicked(self.save_gif)
        self.btn_an = self._button([0.378, 0.048, 0.135, 0.040], "Analysis")
        self.btn_an.on_clicked(self.open_analysis)
        self.status = self.fig.text(0.098, 0.020, "", ha="left", fontsize=11,
                                    color=MUTED)

        # ---- blitting: only the curves + slider handles redraw per event
        for s in self.sliders.values():
            s.drawon = False
            s.valtext.set_visible(False)          # the entry box shows the value
        self._bg = None
        self._anim = [ln for pair in self.lines for ln in pair] + [self.title]
        for s in self.sliders.values():
            self._anim += [s.poly, s._handle]
        for art in self._anim:
            art.set_animated(True)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.fig.canvas.mpl_connect("button_release_event", lambda _e: self._notify())
        self.fig.canvas.mpl_connect("resize_event", self._on_resize)
        self._apply_preset("PDG", quiet=True)

    # ---- listeners -----------------------------------------------------
    def add_listener(self, fn):
        self._listeners.append(fn)

    def _notify(self):
        self._sync_entries()
        self._rescale_app()
        self._bg = None
        self.fig.canvas.draw_idle()      # refresh non-animated widgets + re-cache bg
        for fn in list(self._listeners):
            fn()

    # ---- styled widget helpers -----------------------------------------
    def _header(self, x, y, text):
        self.fig.text(x, y, text, fontsize=14, weight="bold", color=INK, va="bottom")

    def _button(self, rect, text):
        b = Button(self.fig.add_axes(rect), text, color="#FFFFFF", hovercolor="#DCE6F2")
        b.label.set_fontsize(13); b.label.set_color(INK)
        for sp in b.ax.spines.values():
            sp.set_color("#C7CCD4")
        return b

    def _add_sliders(self, specs, x, w, y0, ex, ew, unc_x):
        y = y0
        for label, key, vmin, vmax, init, scale, fmt in specs:
            ax = self.fig.add_axes([x, y + 0.005, w, 0.016])
            s = Slider(ax, label, vmin, vmax, valinit=init, valfmt=fmt,
                       color=ACCENT, track_color=TRACK,
                       handle_style={"facecolor": "white", "edgecolor": ACCENT,
                                     "size": 11})
            s.label.set_fontsize(13); s.label.set_color(INK)
            s.scale = scale
            s.on_changed(lambda _v: self.update())
            self.sliders[key] = s
            self.fmt[key] = fmt

            tb = _style_entry(self.fig.add_axes([ex, y, ew, 0.026]), fmt % init)
            tb.on_submit(lambda t, k=key: self._on_entry(k, t))
            self.entries[key] = tb

            if unc_x is not None:
                self.unc[key] = self.fig.text(unc_x, y + 0.006, "", fontsize=10,
                                              color=MUTED, va="bottom")
            y -= 0.046

    # ---- blitting ------------------------------------------------------
    def _on_draw(self, _event):
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_anim()

    def _draw_anim(self):
        for art in self._anim:
            art.axes.draw_artist(art)

    def _blit(self):
        cv = self.fig.canvas
        if self._bg is None:
            cv.draw_idle(); return
        cv.restore_region(self._bg)
        self._draw_anim()
        cv.blit(self.fig.bbox)
        cv.flush_events()

    # ---- state ---------------------------------------------------------
    def _vals(self):
        return {k: s.val * s.scale for k, s in self.sliders.items()}

    def energy(self):
        return jnp.linspace(self.emin, self.emax, NE)

    def _state(self, **override):
        v = self._vals(); v.update(override)
        params = build_params(v["s12"], v["s13"], v["s23"], v["dm21"], v["dm32"],
                              v["dcp"], self.radio_mo.value_selected)
        nsi = build_nsi(v["eps_ee"], v["eps_emu"], v["eps_etau"], v["eps_mutau"])
        return params, v["L"], nsi

    def update(self, *_):
        params, L, nsi = self._state()
        e = self.energy()
        enp = np.asarray(e)
        data = compute(params, L, nsi, e, self.init_flav)
        clipped = False
        for i, (ln, lb) in enumerate(self.lines):
            nu, nub = data[2 * i], data[2 * i + 1]
            ln.set_data(enp, nu); lb.set_data(enp, nub)
            ys = _finite(nu, nub)
            bot, top = self.axes[i].get_ylim()
            if ys.size and (ys.max() > top or ys.min() < bot):
                clipped = True                 # every panel autoscales
        self.axes[0].set_xlim(self.emin, self.emax)
        self.title.set_text(
            fr"{self.radio_mo.value_selected} ordering,  "
            fr"$\delta_{{CP}}={self.sliders['dcp'].val:.2f}$,  "
            fr"$L={self.sliders['L'].val:.0f}$ km")
        # axis limits live in the blit background, so a rescale needs a full draw
        if clipped:
            self._rescale_app()
            self.fig.canvas.draw_idle()
            return
        self._blit()

    def _rescale_app(self):
        chans = panels.channel_list(self.init_flav)
        for ax, (ln, lb), fout in zip(self.axes, self.lines, chans):
            ys = _finite(ln.get_ydata(), lb.get_ydata())
            if not ys.size:
                continue
            if fout == self.init_flav:                       # survival
                ax.set_ylim(*_nice_span(float(ys.min()), float(ys.max())))
            else:                                            # appearance
                ax.set_ylim(0, _nice_top(float(ys.max()) * 1.12))

    def _refresh(self):
        """Recompute, rescale every panel, and force a full redraw.

        Axis limits live in the cached blit background, so after a rescale the
        cache must be dropped -- otherwise curves get drawn against a stale
        frame. Used by every non-drag input (entry boxes, presets, radios);
        dragging stays on the fast blit path and rescales on mouse-release.
        """
        self.update()
        self._rescale_app()
        self._bg = None
        self.fig.canvas.draw_idle()

    def _on_resize(self, _event=None):
        """Panel labels are sized in points, so they must be refitted when the
        window changes; the cached blit background is invalid afterwards."""
        fs = panels.fit_labels(self.fig, self.axes, self.init_flav)
        self.title.set_fontsize(max(9.0, min(20.0, fs + 5)))
        # the header is left-aligned and the title centred, so on a narrow
        # window the long header runs into it -- shorten rather than overlap
        self.header.set_fontsize(max(9.0, min(17.0, fs + 3)))
        self.header.set_text("jaxnu  ·  neutrino oscillation explorer"
                             if self.fig.get_figwidth() >= 11.5 else "jaxnu")
        self._bg = None

    def _set_flavour(self, label):
        self.init_flav = {r"$\nu_e$": Flavor.E, r"$\nu_\mu$": Flavor.MU,
                          r"$\nu_\tau$": Flavor.TAU}[label]
        panels.apply_mode(self.axes, self.lines, self.init_flav)
        self._refresh()

    # ---- entry boxes / presets ----------------------------------------
    def _sync_entries(self):
        self._syncing = True
        for k, tb in self.entries.items():
            txt = self.fmt[k] % self.sliders[k].val
            if tb.text != txt:
                tb.set_val(txt)
        self._syncing = False

    def _on_entry(self, key, text):
        if self._syncing:
            return
        try:
            v = float(text)
        except ValueError:
            self._sync_entries(); return
        s = self.sliders[key]
        s.set_val(min(max(v, s.valmin), s.valmax))     # triggers update()
        self._sync_entries()
        self._refresh()

    def _on_energy(self):
        if self._syncing:
            return
        try:
            lo, hi = float(self.e_lo.text), float(self.e_hi.text)
        except ValueError:
            return
        if hi <= lo or lo < 0:
            self._flash("E range must satisfy 0 <= min < max"); return
        self.emin, self.emax = lo, hi
        self._refresh()
        for fn in list(self._listeners):
            fn()

    def _apply_preset(self, name, quiet=False):
        for key, (val, unc) in PRESETS[name].items():
            if key in self.sliders:
                s = self.sliders[key]
                s.set_val(min(max(val, s.valmin), s.valmax))
            if key in self.unc:
                self.unc[key].set_text(_unc_str(unc))
        self._sync_entries()
        self._refresh()
        if not quiet:
            self._flash(f"loaded {name} values")

    def open_analysis(self, *_):
        if self.analysis is not None and plt.fignum_exists(self.analysis.fig.number):
            self._flash("analysis window already open"); return
        self.analysis = AnalysisWindow(self)
        try:
            self.analysis.fig.show()
        except Exception:
            pass
        self._flash("analysis window opened")

    # ---- exports -------------------------------------------------------
    def save_png(self, *_):
        FIGDIR.mkdir(exist_ok=True)
        out = FIGDIR / "gui_snapshot"
        clean = self._panels_only()
        clean.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
        clean.savefig(f"{out}.pdf", bbox_inches="tight")
        plt.close(clean)
        self._flash(f"saved {out.name}.png / .pdf")

    def _panels_only(self):
        fig = plt.figure(figsize=(8.4, 9.2), dpi=100)
        axes, lines = panels.build_panels(fig, self.init_flav,
                                          top=0.92, bot=0.09, fs=14)
        params, L, nsi = self._state()
        e = self.energy(); enp = np.asarray(e)
        data = compute(params, L, nsi, e, self.init_flav)
        for i, (ln, lb) in enumerate(lines):
            nu, nub = data[2 * i], data[2 * i + 1]
            ln.set_data(enp, nu); lb.set_data(enp, nub)
            ys = _finite(nu, nub)
            if not ys.size:
                continue
            if i:
                axes[i].set_ylim(0, _nice_top(float(ys.max()) * 1.12))
            else:
                axes[i].set_ylim(*_nice_span(float(ys.min()), float(ys.max())))
        axes[0].set_xlim(self.emin, self.emax)
        axes[0].set_title(fr"{self.radio_mo.value_selected} ordering,  "
                          fr"$\delta_{{CP}}={self.sliders['dcp'].val:.2f}$,  "
                          fr"$L={self.sliders['L'].val:.0f}$ km")
        return fig

    def _sweep_plan(self):
        sel = self.radio_sweep.value_selected
        if sel.startswith(r"$\delta"):
            return "dcp", np.linspace(0, 2 * np.pi, 73)[:-1], \
                lambda v: fr"$\delta_{{CP}} = {v/np.pi:.2f}\,\pi$", False
        if sel.startswith(r"$\sin"):
            return "s23", np.linspace(0.35, 0.65, 55), \
                lambda v: fr"$\sin^2\theta_{{23}} = {v:.3f}$", True
        if sel.startswith(r"$\Delta"):
            return "dm32", np.linspace(2.30e-3, 2.70e-3, 55), \
                lambda v: fr"$\Delta m^2_{{32}} = {v*1e3:.2f}\times10^{{-3}}$", True
        if sel == r"$\epsilon_{e\mu}$":
            return "eps_emu", np.linspace(-0.3, 0.3, 55), \
                lambda v: fr"$\epsilon_{{e\mu}} = {v:+.2f}$", True
        if sel == r"$\epsilon_{e\tau}$":
            return "eps_etau", np.linspace(-0.3, 0.3, 55), \
                lambda v: fr"$\epsilon_{{e\tau}} = {v:+.2f}$", True
        return "ordering", ["Normal", "Inverted"], lambda v: f"{v} ordering", False

    def save_gif(self, *_):
        key, vals, labeller, pingpong = self._sweep_plan()
        order = (list(range(len(vals))) + list(range(len(vals) - 2, 0, -1))
                 if pingpong else list(range(len(vals))))
        if key == "ordering":
            order = [0] * 24 + [1] * 24

        fig = plt.figure(figsize=(8.4, 9.2), dpi=100)
        axes, lines = panels.build_panels(fig, self.init_flav,
                                          top=0.92, bot=0.09, fs=14)
        axes[0].set_xlim(self.emin, self.emax)
        title = axes[0].set_title("")
        e = self.energy(); enp = np.asarray(e)

        def frame(k):
            v = vals[order[k]]
            if key == "ordering":
                vv = self._vals()
                params = build_params(vv["s12"], vv["s13"], vv["s23"], vv["dm21"],
                                      vv["dm32"], vv["dcp"], v)
                _, L, nsi = self._state()
            else:
                params, L, nsi = self._state(**{key: v})
            data = compute(params, L, nsi, e, self.init_flav)
            for i, (ln, lb) in enumerate(lines):
                ln.set_data(enp, data[2 * i]); lb.set_data(enp, data[2 * i + 1])
            title.set_text(labeller(v))
            return [ln for pair in lines for ln in pair] + [title]

        peaks = [0.0] * len(lines)          # pre-scan so the axes never jump
        lows = [1.0] * len(lines)
        for k in range(len(order)):
            frame(k)
            for i, (ln, lb) in enumerate(lines):
                ys = _finite(ln.get_ydata(), lb.get_ydata())
                if ys.size:
                    peaks[i] = max(peaks[i], float(ys.max()))
                    lows[i] = min(lows[i], float(ys.min()))
        axes[0].set_ylim(*_nice_span(lows[0], peaks[0]))
        for i in range(1, len(lines)):
            axes[i].set_ylim(0, _nice_top(peaks[i] * 1.12))

        self._flash("rendering GIF...")
        anim = FuncAnimation(fig, frame, frames=len(order), interval=1000 / FPS,
                             blit=False)
        FIGDIR.mkdir(exist_ok=True)
        out = FIGDIR / f"gui_sweep_{key}.gif"
        anim.save(out, writer=PillowWriter(fps=FPS), dpi=100,
                  savefig_kwargs={"bbox_inches": None})
        plt.close(fig)
        self._flash(f"saved {out.name}")

    def _flash(self, msg):
        print(msg)
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()


class AnalysisWindow:
    """Opened on demand from the main GUI. Three autodiff-powered modes."""
    MODES = ("Sensitivity", "Uncertainty band", "Fit contour")
    PARAMS = [(r"$\delta_{CP}$", "dcp"), (r"$\sin^2\theta_{23}$", "s23"),
              (r"$\Delta m^2_{32}$", "dm32"), (r"$\sin^2\theta_{13}$", "s13")]
    PAIRS = [(r"$\theta_{23}$ – $\Delta m^2_{32}$", ("s23", "dm32")),
             (r"$\theta_{23}$ – $\delta_{CP}$",     ("s23", "dcp")),
             (r"$\theta_{13}$ – $\delta_{CP}$",     ("s13", "dcp")),
             (r"$\Delta m^2_{32}$ – $\delta_{CP}$", ("dm32", "dcp"))]
    SIGMA = {"s13": 0.00065, "s23": 0.05, "dm32": 5e-5, "dcp": 0.7}
    PLAB = {"dcp": r"$\delta_{CP}$", "s23": r"$\sin^2\theta_{23}$",
            "dm32": r"$\Delta m^2_{32}$", "s13": r"$\sin^2\theta_{13}$"}

    def __init__(self, gui):
        self.gui = gui
        self.mode = self.MODES[0]
        self.sens_param = "dcp"
        self.pair = self.PAIRS[0][1]
        self.fig = plt.figure(figsize=(8.6, 9.6), dpi=100, facecolor=UI_BG)
        self.fig.text(0.03, 0.985, "Analysis  (uses jaxnu autodiff)",
                      fontsize=14, weight="bold", va="top", color=INK)

        self.fig.text(0.04, 0.945, "Mode", fontsize=11, weight="bold", color=INK)
        self.radio_mode = _style_radio(self.fig.add_axes([0.03, 0.845, 0.29, 0.095]),
                                       self.MODES, fontsize=11)
        self.radio_mode.on_clicked(self._set_mode)

        self.fig.text(0.36, 0.945, "Sensitivity param", fontsize=11, weight="bold",
                      color=INK)
        self.radio_param = _style_radio(self.fig.add_axes([0.35, 0.845, 0.29, 0.095]),
                                        [p[0] for p in self.PARAMS], fontsize=11)
        self.radio_param.on_clicked(self._set_param)

        self.fig.text(0.68, 0.945, "Contour axes", fontsize=11, weight="bold",
                      color=INK)
        self.radio_pair = _style_radio(self.fig.add_axes([0.67, 0.845, 0.31, 0.095]),
                                       [p[0] for p in self.PAIRS], fontsize=11)
        self.radio_pair.on_clicked(self._set_pair)

        self.status = self.fig.text(0.03, 0.02, "", fontsize=10, color=MUTED)
        self._axarea = (0.13, 0.09, 0.82, 0.70)
        self._axes = []
        self.gui.add_listener(self.refresh)
        self._build_and_draw()

    # ---- callbacks -----------------------------------------------------
    def _set_mode(self, label):
        self.mode = label; self._build_and_draw()

    def _set_param(self, label):
        self.sens_param = dict(self.PARAMS)[label]
        if self.mode == "Sensitivity":
            self._build_and_draw()

    def _set_pair(self, label):
        self.pair = dict(self.PAIRS)[label]
        if self.mode == "Fit contour":
            self._build_and_draw()

    def refresh(self):
        if not plt.fignum_exists(self.fig.number):     # window was closed
            return
        self._build_and_draw()

    # ---- axes ----------------------------------------------------------
    def _clear(self):
        for ax in self._axes:
            ax.remove()
        self._axes = []

    @staticmethod
    def _style(*axes):
        for ax in axes:
            ax.set_facecolor("white")
            ax.grid(color="#DDE1E7", lw=0.8, alpha=1.0)
            for sp in ax.spines.values():
                sp.set_color("#C7CCD4")
            ax.tick_params(colors=MUTED, labelcolor=INK)
            ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)

    def _n_panels(self, n):
        self._clear()
        l, b, h_tot = self._axarea[0], self._axarea[1], self._axarea[3]
        w = self._axarea[2]
        gap = 0.03
        h = (h_tot - gap * (n - 1)) / n
        axes = []
        for i in range(n):
            ax = self.fig.add_axes([l, b + h_tot - h - i * (h + gap), w, h],
                                   sharex=axes[0] if axes else None)
            axes.append(ax)
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)
        self._style(*axes)
        self._axes = axes
        return axes

    def _one_panel(self):
        self._clear()
        l, b, w, h = self._axarea
        ax = self.fig.add_axes([l, b, w, h])
        self._style(ax)
        self._axes = [ax]
        return ax

    def _build_and_draw(self):
        self.status.set_text("computing...")
        self.fig.canvas.draw_idle()
        try:
            {"Sensitivity": self._draw_sens,
             "Uncertainty band": self._draw_band,
             "Fit contour": self._draw_contour}[self.mode]()
            self.status.set_text("")
        except Exception as exc:
            self.status.set_text(f"error: {exc}")
            print("AnalysisWindow error:", exc)
        self.fig.canvas.draw_idle()

    # ---- the three analyses --------------------------------------------
    def _draw_sens(self):
        base, L, nsi = self.gui._state()
        e = self.gui.energy(); enp = np.asarray(e)
        x0 = self.gui._vals()[self.sens_param]
        init = self.gui.init_flav
        chans = panels.channel_list(init)
        axes = self._n_panels(len(chans))
        for ax, fout in zip(axes, chans):
            d = np.asarray(sens_jac(self.sens_param, False, init, fout)(
                float(x0), base, float(L), nsi, e))
            ax.plot(enp, d, color=panels.CHAN_COL[fout][0])
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_ylabel(
                fr"$\partial P({panels.TEX[init]}\to{panels.TEX[fout]})"
                fr"/\partial${self.PLAB[self.sens_param]}", fontsize=11)
        axes[0].set_xlim(self.gui.emin, self.gui.emax)
        axes[-1].set_xlabel(r"$E_\nu$  (GeV)")
        axes[0].set_title(f"Sensitivity to {self.PLAB[self.sens_param]}")

    def _draw_band(self):
        base, L, nsi = self.gui._state()
        e = self.gui.energy(); enp = np.asarray(e)
        v = self.gui._vals()
        vec = jnp.array([v["s13"], v["s23"], v["dm32"], v["dcp"]])
        sig = np.array([self.SIGMA[k] for k in ("s13", "s23", "dm32", "dcp")])
        init = self.gui.init_flav
        chans = panels.channel_list(init)
        axes = self._n_panels(len(chans))
        for i, (ax, fout) in enumerate(zip(axes, chans)):
            col = panels.CHAN_COL[fout][0]
            J = np.asarray(band_jac(False, init, fout)(vec, base, float(L), nsi, e))
            Pc = np.asarray(_P(base, float(L), nsi, e, False, init, fout))
            sP = np.sqrt((J ** 2) @ (sig ** 2))
            ax.plot(enp, Pc, color=col)
            ax.fill_between(enp, np.clip(Pc - sP, 0, None), Pc + sP,
                            color=col, alpha=0.25, lw=0)
            lo_y, hi_y = _finite(Pc - sP), _finite(Pc + sP)
            if lo_y.size and hi_y.size:
                ax.set_ylim(*(_nice_span(float(lo_y.min()), float(hi_y.max()))
                              if fout == init
                              else (0, _nice_top(float(hi_y.max()) * 1.12))))
            ax.set_ylabel(fr"$P({panels.TEX[init]}\to{panels.TEX[fout]})$",
                          fontsize=11)
        axes[0].set_xlim(self.gui.emin, self.gui.emax)
        axes[-1].set_xlabel(r"$E_\nu$  (GeV)")
        axes[0].set_title(r"$\pm1\sigma$ band (Jacobian propagation)")

    def _draw_contour(self):
        base, L, nsi = self.gui._state()
        e = self.gui.energy()
        v = self.gui._vals()
        xk, yk = self.pair
        ch = panels.channel_list(self.gui.init_flav)
        xs, ys, D, truth, fit = contour_data(base, float(L), nsi, e,
                                             xk, yk, v[xk], v[yk],
                                             init=ch[0], other=ch[1])
        xlab, _, _, xsc = CONTOUR_RANGE[xk]
        ylab, _, _, ysc = CONTOUR_RANGE[yk]
        ax = self._one_panel()
        cs = ax.contour(xs * xsc, ys * ysc, D, levels=[2.30, 6.18, 11.83],
                        colors=[BLUE_D, GOLD_D, "#B04A6A"], linewidths=2)
        ax.clabel(cs, fmt={2.30: "68%", 6.18: "90%", 11.83: "99%"}, fontsize=10)
        ax.plot(truth[0] * xsc, truth[1] * ysc, "*", color="k", ms=16, label="truth")
        ax.plot(fit[0] * xsc, fit[1] * ysc, "o", color="#B04A6A", ms=9,
                label="grad fit")
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.legend(frameon=False, loc="best")
        ax.set_title("Asimov contours — illustrative only", fontsize=15)


def main():
    app = OscGUI()          # keep a reference so widget callbacks stay alive
    plt.show()
    return app


if __name__ == "__main__":
    main()
