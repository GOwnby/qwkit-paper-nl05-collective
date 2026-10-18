"""NL-05 benchmark: Zaluzny, Phys. Rev. B 51, 9757 (1995). GaAs asymmetric step well (total 125 A, narrow well 50 A,
barriers 350 meV, step height V), parabolic m* = 0.066, eps_inf = 11.1, N_s = 1e12 cm^-2, Gamma = 5 meV, flat band,
only the ground subband occupied, three subbands. Published values: one-electron double resonance at V = 115.6 meV,
hw = 102 meV; diagonal approximation (energy substitution) at V = 102.2 meV, hw = 102.3 meV; with the 1-2/2-3
coupling the maximum |chi2|^2 is at V = 97.2 meV, hw = 102.2 meV and is 2.5 times the diagonal-approximation maximum.
We compute |chi2|^2 spectra vs V with the qwkit TDH kernel restricted to (a) no kernel, (b) self blocks only,
(c) self + cross blocks (Zaluzny's model: no diagonal/permanent-dipole Hartree terms), (d) the full kernel.
Writes zaluzny95.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import core, tdh
from masks import blockmask

M = 0.066; EPS = 11.1; GAM = 0.005; NS = 1e12
core.add_material("z_well", 0.0, M, 10.0); core.add_material("z_bar", 0.350, M, 10.0)


def states(Vstep):
    core.add_material("z_step", Vstep, M, 10.0)
    L = [(50.0, "z_well"), (75.0, "z_step")]
    z, V0, m, Eg, act = core.build(L, pad=150.0, dz=0.5, pad_material="z_bar")
    E, psi = core.solve_states(z, V0, m, Eg, nstates=3, nonparabolic=False)
    psi = core.orthonormalize(z, psi)
    return z, E, psi, 0.5 * (act[0] + act[1]), act


def model(Vstep, kind):
    z, E, psi, zc, act = states(Vstep)
    N = NS / ((act[1] - act[0]) * 1e-8)
    m = tdh.TDH(z, psi, E, [1.0, 0, 0], NS, N, gamma=GAM, eps_r=EPS, zc=zc, kernel=(kind != "bare"))
    if kind in ("self", "self+cross"):
        keep = blockmask(3, "self") | (blockmask(3, "cross") if kind == "self+cross" else False)
        m.S = np.where(keep, m.S, 0.0); m._build()
    return m, E


if __name__ == "__main__":
    hws = np.arange(0.090, 0.1150001, 0.0002)
    Vs = np.arange(0.085, 0.1300001, 0.0008)
    out = dict(V=Vs.tolist(), hw=hws.tolist(), res={})
    for kind in ["bare", "self", "self+cross", "full"]:
        best = None; grid = []
        for V in Vs:
            m, E = model(V, kind)
            c = np.array([abs(m.chi2(h)) ** 2 for h in hws])
            grid.append(float(c.max()))
            if best is None or c.max() > best[0]:
                best = (float(c.max()), float(V), float(hws[c.argmax()]), float(1e3 * (E[1] - E[0])), float(1e3 * (E[2] - E[0])))
        out["res"][kind] = dict(max=best[0], V_meV=best[1] * 1e3, hw_meV=best[2] * 1e3, E21=best[3], E31=best[4], peak_vs_V=grid)
        print(f"{kind:10s}: max |chi2|^2 at V = {best[1]*1e3:.1f} meV, hw = {best[2]*1e3:.1f} meV (E21 {best[3]:.1f}, E31/2 {best[4]/2:.1f} meV)", flush=True)
    b = out["res"]["self"]["max"]
    for k in out["res"]:
        print(f"   max |chi2|^2 relative to the self-only (diagonal-approximation) maximum: {k} {out['res'][k]['max']/b:.2f}")
    z, E, psi, zc, act = states(0.110)
    Z = core.dipoles(z, psi)
    S = tdh.hartree_kernel(z, psi)
    K = tdh.QE * NS * 1e4 / (tdh.EPS0 * EPS)
    E21 = E[1] - E[0]
    a22 = 2 * K * S[0, 1, 0, 1] / E21; a32b = 2 * K * Z[0, 1] / Z[1, 2] * S[1, 2, 0, 1] / E21
    out["V110"] = dict(E21=E21 * 1e3, E31=(E[2] - E[0]) * 1e3, P_A6=float((Z[0, 1] * Z[0, 2] * Z[1, 2]) ** 2), alpha22=a22, alpha32bar=a32b)
    print(f"V = 110 meV: E21 {E21*1e3:.1f}, E31/2 {(E[2]-E[0])*500:.1f} meV, P = (z21 z31 z32)^2 = {out['V110']['P_A6']:.2e} A^6, "
          f"alpha22 = {a22:.3f}, alpha32bar = {a32b:.3f} (kappa_exc = {a32b/a22:.2f})")
    json.dump(out, open(os.path.join(here, "zaluzny95.json"), "w"))
