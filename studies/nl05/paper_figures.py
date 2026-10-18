"""Figures for the NL-05 manuscript, drawn only from the study JSON outputs. Writes ../../paper_nl05/figures/*.pdf."""
import os, json
here = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

outdir = os.path.join(here, "..", "..", "paper_nl05", "figures"); os.makedirs(outdir, exist_ok=True)
BLUE, ORANGE, AQUA, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#8a4fbf"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.0, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "lines.linewidth": 1.5,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
SP = json.load(open(os.path.join(here, "spectra.json")))
PW = json.load(open(os.path.join(here, "pathway.json")))
DY = json.load(open(os.path.join(here, "dynamic.json")))
KG = json.load(open(os.path.join(here, "kernel_geometry.json")))
DC = json.load(open(os.path.join(here, "double_count.json")))
DE = json.load(open(os.path.join(here, "decompose.json")))
STY = {"bare": (MUTED, ":", "bare"), "ando": (ORANGE, "--", "Ando substitution"), "lf": (AQUA, "-.", "local field ($\\kappa=1$)"),
       "tdh": (BLUE, "-", "TDH")}


def panel_label(ax, s, x=-0.17, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", color=INK)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)


def get(name, ns):
    return next(o for o in SP if o["name"] == name and abs(o["ns"] / ns - 1) < 1e-6)


def fig_geometry():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.32))
    for ax, k, lab, tag in zip(axs, [0, 1], "ab", ["ADQW 64/10/29 Å", "triple well, 55 kV/cm"]):
        g = KG[k]; z = np.array(g["z"]) / 10
        sg = -1 if min(g["F01"]) < -max(g["F01"]) else 1
        ax.plot(z, sg * np.array(g["F01"]) / g["z01"] * 10, color=BLUE, label="$F_{12}(z)/z_{12}$")
        ax.plot(z, sg * np.array(g["F12"]) / g["z12"] * 10, color=ORANGE, ls="--", label="$F_{23}(z)/z_{23}$")
        ax.set_xlabel("$z$ (nm)"); ax.set_ylabel("normalised $F_{ij}$ (nm$^{-1}$)")
        ax.set_title(f"{tag}: $L_{{12}}$={g['L01_A']/10:.1f} nm, $L_{{23}}$={g['L12_A']/10:.1f} nm, "
                     f"$c$={abs(g['c']):.2f}, $\\kappa$={g['kappa']:.2f}", fontsize=7.2)
        clean(ax); panel_label(ax, lab); ax.legend(frameon=False, loc="upper left")
    fig.savefig(os.path.join(outdir, "fig1_geometry.pdf")); plt.close(fig)


def fig_spectra():
    cases = [("ADQW", 1e12, "ADQW, $10^{12}$ cm$^{-2}$"), ("ADQW", 4e12, "ADQW, $4\\times10^{12}$ cm$^{-2}$"),
             ("TW", 2e12, "triple well, $2\\times10^{12}$ cm$^{-2}$")]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.36))
    for ax, (nm, ns, t), lab in zip(axs, cases, "abc"):
        o = get(nm, ns); hw = np.array(o["hw"]) * 1e3
        for k in ["bare", "ando", "lf", "tdh"]:
            c, ls, l = STY[k]
            ax.plot(hw, np.abs(np.array(o[k]) @ [1, 1j]) / 1e3, color=c, ls=ls, label=l)
        ax.set_xlabel("pump photon energy (meV)"); ax.set_ylabel("$|\\chi^{(2)}_{zzz}|$ (nm/V)"); ax.set_title(t)
        ax.set_xlim(100, 150); clean(ax); panel_label(ax, lab)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.09))
    fig.savefig(os.path.join(outdir, "fig2_spectra.pdf")); plt.close(fig)


def fig_ratio():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.32))
    for nm, col, mk in [("ADQW", BLUE, "o"), ("TW", ORANGE, "s")]:
        os_ = sorted([o for o in SP if o["name"] == nm], key=lambda o: o["ns"])
        ns = np.array([o["ns"] for o in os_])
        pk = lambda o, k: np.abs(np.array(o[k]) @ [1, 1j]).max()
        axs[0].plot(ns, [pk(o, "tdh") / pk(o, "ando") for o in os_], color=col, marker=mk, ms=3.5, label=f"{nm}: TDH / Ando")
        axs[0].plot(ns, [pk(o, "lf") / pk(o, "ando") for o in os_], color=col, marker=mk, ms=3.5, ls="--", mfc="white",
                    label=f"{nm}: local field / Ando")
        axs[1].plot(ns, [max(o["a1_tdh"]) / max(o["a1_ando"]) for o in os_], color=col, marker=mk, ms=3.5, label=f"{nm}")
    for ax in axs:
        ax.set_xscale("log"); ax.set_xlabel("sheet density $n_s$ (cm$^{-2}$)"); clean(ax); ax.axhline(1, color=INK2, lw=0.6)
    axs[0].set_ylabel("peak $|\\chi^{(2)}|$ ratio"); axs[1].set_ylabel("peak $\\alpha_\\omega$: TDH / Ando")
    axs[0].legend(frameon=False, fontsize=6.3, loc="lower left"); axs[1].legend(frameon=False)
    panel_label(axs[0], "a"); panel_label(axs[1], "b")
    fig.savefig(os.path.join(outdir, "fig3_ratio.pdf")); plt.close(fig)


def fig_pathway():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.32))
    for o, col in zip(PW[:3], [AQUA, BLUE, ORANGE]):
        hw = np.array([r["hw"] for r in o["rows"]]) * 1e3
        F = np.array([complex(*r["Fexc"]) for r in o["rows"]]); Fr = np.array([complex(*r["Fexc_rwa"]) for r in o["rows"]])
        lab = f"$n_s$={o['ns']/1e12:g}$\\times10^{{12}}$"
        axs[0].plot(hw, F.real, color=col, label=lab); axs[0].plot(hw, Fr.real, color=col, ls=":", lw=1.1)
        axs[0].plot(hw, F.imag, color=col, ls="--"); axs[0].plot(hw, Fr.imag, color=col, ls=":", lw=1.1)
    axs[0].set_ylabel("$F_\\mathrm{exc}$: Re (solid), Im (dashed)"); axs[0].set_xlabel("pump photon energy (meV)")
    R = DE["roles"]; modes = [("full", BLUE, "full TDH"), ("no_propagation", AQUA, "cross block only in source"),
                              ("no_source", ORANGE, "cross block only in propagation"), ("none", MUTED, "no cross block")]
    x = np.arange(len(R)); wd = 0.2
    for j, (k, c, l) in enumerate(modes):
        axs[1].bar(x + (j - 1.5) * wd, [r["ratio"][k] for r in R], wd, color=c, label=l)
    axs[1].set_xticks(x); axs[1].set_xticklabels([f"{r['T']:.0f} K\n{r['ns']/1e12:g}e12\n$\\rho_{{22}}$={r['rho'][1]:.2f}" for r in R], fontsize=6.5)
    axs[1].set_ylabel("peak $|\\chi^{(2)}|$: TDH / Ando"); axs[1].axhline(1, color=INK2, lw=0.6); axs[1].set_ylim(0, 1.9)
    axs[1].legend(frameon=False, fontsize=5.8, loc="upper left", ncol=2)
    for ax, l in zip(axs, "ab"):
        clean(ax); panel_label(ax, l)
    axs[0].legend(frameon=False, fontsize=6.3, loc="upper left"); axs[0].text(0.03, 0.04, "dotted: RWA formula", transform=axs[0].transAxes, fontsize=6.5, color=INK2)
    fig.savefig(os.path.join(outdir, "fig4_pathway.pdf")); plt.close(fig)


def fig_dynamic():
    D2 = json.load(open(os.path.join(here, "dynamic2.json")))
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.35), gridspec_kw=dict(wspace=0.45))
    for ns, col in [(5e11, BLUE), (2e12, ORANGE)]:
        rs = sorted([r for r in DY if abs(r["ns"] / ns - 1) < 1e-6], key=lambda r: r["I_MWcm2"])
        I = np.array([r["I_MWcm2"] for r in rs])
        ct = np.array([abs(complex(*r["chi2_tdh"])) for r in rs]) / 1e3; cf = np.array([abs(complex(*r["chi2_frozen"])) for r in rs]) / 1e3
        qs = np.array([abs(complex(*q["chi2"])) for q in D2["quasi_static"][str(ns)]]) / 1e3
        br = sorted([o for o in D2["bare"] if abs(o["ns"] / ns - 1) < 1e-6], key=lambda o: o["I_MWcm2"])
        bratio = np.array([abs(complex(*o["chi2_rt"])) / abs(complex(*o["chi2_rate"])) for o in br])
        lab = f"{ns/1e12:g}$\\times10^{{12}}$"
        axs[0].loglog(I, ct, color=col, marker="o", ms=3, label=f"TDH, {lab}")
        axs[0].loglog(I, cf, color=col, ls="--", marker="s", ms=3, mfc="white", label=f"frozen shift")
        axs[0].loglog(I, qs, color=col, ls=":", marker="^", ms=3, mfc="white", label=f"quasi-static energies")
        axs[1].semilogx(I, ct / cf, color=col, marker="o", ms=3, label=f"TDH / frozen, {lab}")
        axs[1].semilogx(I, qs / cf, color=col, ls=":", marker="^", ms=3, mfc="white", label="quasi-static / frozen")
        axs[1].semilogx(I, bratio, color=col, ls="-.", marker="d", ms=3, mfc="white", label="bare: real time / rate")
        pr_ = D2["probe"][str(ns)]["runs"]
        for k, p in enumerate(pr_):
            for e, h in p["peaks"]:
                axs[2].semilogx(p["I_MWcm2"], e, color=col, marker="o", ms=2 + 3 * h, ls="none", label=lab if (k == 0 and h == 1.0) else None)
        axs[2].semilogx([p["I_MWcm2"] for p in pr_], [p["E12_static_meV"] for p in pr_], color=col, ls=":", lw=1.0)
    axs[0].set_ylabel("$|\\chi^{(2)}_\\mathrm{eff}|$ (nm/V)"); axs[1].set_ylabel("ratio of $|\\chi^{(2)}_\\mathrm{eff}|$")
    axs[2].set_ylabel("pump-band absorption peaks (meV)"); axs[2].set_ylim(98, 137); axs[2].axhline(122.5, color=INK2, lw=0.6, ls="--")
    axs[2].text(0.012, 117.0, "pump, $2\\times10^{12}$", fontsize=6.3, color=INK2)
    for ax, l in zip(axs, "abc"):
        ax.set_xlabel("pump intensity (MW/cm$^2$)"); clean(ax); panel_label(ax, l, x=-0.26)
    axs[0].legend(frameon=False, fontsize=5.4, loc="lower left", ncol=1); axs[1].legend(frameon=False, fontsize=5.2, loc="lower left")
    axs[2].legend(frameon=False, fontsize=6.3, title="$n_s$ (cm$^{-2}$)", title_fontsize=6.3, loc="lower right", ncol=2)
    fig.savefig(os.path.join(outdir, "fig5_dynamic.pdf")); plt.close(fig)


def fig_absorption():
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.2), gridspec_kw=dict(wspace=0.3))
    for ax, ns, lab in zip(axs, [2e12, 4e12], "ab"):
        o = get("ADQW", ns); hw = np.array(o["hw"]) * 1e3
        for k, key in [("bare", "a1_bare"), ("ando", "a1_ando"), ("tdh", "a1_tdh")]:
            c, ls, l = STY[k]; ax.plot(hw, np.array(o[key]) / 100, color=c, ls=ls, label=l)
        ax.set_xlabel("photon energy (meV)"); ax.set_ylabel("$\\alpha$ (cm$^{-1}$)"); ax.set_title(f"ADQW, $n_s$ = {ns/1e12:g}$\\times10^{{12}}$ cm$^{{-2}}$")
        ax.set_xlim(90, 160); clean(ax); panel_label(ax, lab)
    axs[0].legend(frameon=False, loc="upper right")
    fig.savefig(os.path.join(outdir, "fig6_absorption.pdf")); plt.close(fig)


for f in [fig_geometry, fig_spectra, fig_ratio, fig_pathway, fig_dynamic, fig_absorption]:
    f()
print("figures written to", os.path.abspath(outdir))
