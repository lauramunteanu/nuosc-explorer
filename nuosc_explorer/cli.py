"""Command-line entry point.

    nuosc-explorer              # open the interactive GUI (needs a display)
    nuosc-explorer --snapshot   # headless: render figures/gui_snapshot.png|pdf
"""
import argparse
import os


def main(argv=None):
    ap = argparse.ArgumentParser(prog="nuosc-explorer",
                                 description="Interactive neutrino-oscillation explorer.")
    ap.add_argument("--snapshot", action="store_true",
                    help="render a snapshot headlessly instead of opening the GUI")
    ap.add_argument("--preset", default="PDG",
                    help="parameter set to load (T2K, NOvA, PDG, NuFIT)")
    ap.add_argument("--outdir", default=None, help="output directory for figures")
    args = ap.parse_args(argv)

    if args.outdir:
        os.environ["NUOSC_FIGDIR"] = args.outdir
    if args.snapshot:                     # pick a non-interactive backend first
        import matplotlib
        matplotlib.use("Agg")

    from . import app

    if not args.snapshot:
        return app.main()

    gui = app.OscGUI()
    gui.fig.canvas.draw()
    if args.preset in app.PRESETS:
        gui._apply_preset(args.preset, quiet=True)
    gui.save_png()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
