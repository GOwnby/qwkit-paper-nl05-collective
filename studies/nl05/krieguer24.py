"""NL-05 benchmark: Krieguer & Todorov, arXiv:2412.08297 (2024), Sec. II B (three-subband system, Figs. 11-12).
GaAs/Al0.3Ga0.7As coupled double wells L1/1 nm/L2 with L1 + L2 = 11 nm, three subbands, only the ground subband
populated (no thermal population), identical broadening 10 meV, flat band (they neglect the static Hartree potential
of the donors), dipole-dipole coupling between the 1-2 and 1-3 plasmons (their W^2), 2-3 at its single-particle
energy, static (diagonal) terms averaged away. In qwkit TDH this is the "self + cross" kernel; we also give the bare
(single-particle, non-RWA), self-only (energy substitution) and full kernels.
Band parameters chosen to reproduce their single-particle energies (Fig. 13 labels: E21 ~ 42, 48, 62, 78, 84 meV;
E31 ~ 150-165 meV): conduction-band offset 0.23 eV, m* = 0.067 (well) and 0.092 (barrier), parabolic;
eps_r = 10.9 (GaAs, high frequency).
Their findings to compare: (1) at 1e10 cm^-2 four resonances at E12/2, E12, E13/2, E13; (2) for 3.6/1/7.4 nm the
collective peak at 1.8e12 cm^-2 is 'almost a factor of 3' above the single-particle RWA expression with the same
charge; (3) at 1e12 cm^-2 the peak |chi2| is largest for 3.6/1/7.4 (collective double resonance E~12 = E~13/2),
whereas single-particle theory favours 2.7/1/8.3. Writes krieguer24.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import core, tdh
from masks import blockmask

GAM = 0.010; EPS = 10.9
core.add_material("kt_w", 0.0, 0.067, 10.0); core.add_material("kt_b", 0.23, 0.092, 10.0)
DESIGNS = [(54, 56), (45, 65), (36, 74), (27, 83), (18, 92)]          # L1, L2 in A (barrier 10 A)
# "_nd": permanent (diagonal) dipoles removed from the light coupling, as in their cavity geometry where the static
# terms average away (their chi2 is proportional to d23 and has no z_nn terms). "kt" = their model.
KINDS = ["bare_nd", "self_nd", "kt", "bare", "self+cross", "full"]


def states(L1, L2):
    L = [(float(L1), "kt_w"), (10.0, "kt_b"), (float(L2), "kt_w")]
    z, V0, m, Eg, act = core.build(L, pad=150.0, dz=0.5, pad_material="kt_b")
    E, psi = core.solve_states(z, V0, m, Eg, nstates=3, nonparabolic=False)
    psi = core.orthonormalize(z, psi)
    return z, E, psi, 0.5 * (act[0] + act[1]), act


def model(L1, L2, ns, kind):
    z, E, psi, zc, act = states(L1, L2)
    N = ns / ((act[1] - act[0]) * 1e-8)
    m = tdh.TDH(z, psi, E, [1.0, 0, 0], ns, N, gamma=GAM, eps_r=EPS, zc=zc, kernel=not kind.startswith("bare"))
    base = {"self_nd": "self", "kt": "self+cross"}.get(kind, kind)
    if base in ("self", "self+cross"):
        keep = blockmask(3, "self") | (blockmask(3, "cross") if base == "self+cross" else False)
        m.S = np.where(keep, m.S, 0.0); m._build()
    if kind.endswith("_nd") or kind == "kt":
        m.Z = m.Z - np.diag(np.diag(m.Z))
    return m, E


def spectrum(m, hws):
    return np.array([abs(m.chi2(h)) for h in hws])


if __name__ == "__main__":
    hws = np.arange(0.020, 0.2500001, 0.0005)
    out = dict(hw=hws.tolist(), designs=DESIGNS, single={}, peaks={})
    for L1, L2 in DESIGNS:
        z, E, psi, zc, act = states(L1, L2)
        out["single"][f"{L1}/{L2}"] = [1e3 * (E[1] - E[0]), 1e3 * (E[2] - E[0])]
    for ns in [1e10, 1e11, 5e11, 1e12, 1.8e12]:
        for L1, L2 in DESIGNS:
            key = f"{ns:.1e}|{L1}/{L2}"; out["peaks"][key] = {}
            for kind in KINDS:
                m, E = model(L1, L2, ns, kind)
                c = spectrum(m, hws)
                i = int(np.argmax(c))
                # local maxima (resonance positions) above 1% of the peak
                loc = [float(hws[j]) for j in range(1, len(c) - 1) if c[j] > c[j - 1] and c[j] > c[j + 1] and c[j] > 0.01 * c[i]]
                out["peaks"][key][kind] = dict(peak=float(c[i] / ns), hw=float(hws[i]), maxima_meV=[1e3 * x for x in loc])
                if L1 == 36 and ns in (1e10, 1.8e12):
                    out.setdefault("spectra", {})[f"{ns:.1e}|{kind}"] = (c / ns).tolist()
        r = {f"{L1}/{L2}": {k: out["peaks"][f"{ns:.1e}|{L1}/{L2}"][k]["peak"] for k in KINDS} for L1, L2 in DESIGNS}
        best = {k: max(r, key=lambda d: r[d][k]) for k in KINDS}
        p = out["peaks"][f"{ns:.1e}|36/74"]
        print(f"ns {ns:.1e}: best design {best}", flush=True)
        print("   3.6/1/7.4 peak relative to bare_nd: " + ", ".join(f"{k} {p[k]['peak']/p['bare_nd']['peak']:.2f} at {p[k]['hw']*1e3:.1f} meV" for k in KINDS), flush=True)
        print("   peak (per ns) by design, kt vs bare_nd: " + ", ".join(f"{d}: {r[d]['kt']/r['36/74']['kt']:.2f}/{r[d]['bare_nd']/r['36/74']['bare_nd']:.2f}" for d in r), flush=True)
        if ns in (1e10, 1.8e12):
            for k in ("bare_nd", "kt"):
                print(f"   3.6/1/7.4 maxima ({k}):", [round(x, 1) for x in p[k]["maxima_meV"]], flush=True)
    json.dump(out, open(os.path.join(here, "krieguer24.json"), "w"))
