"""NL-05 study 1: second-order response with and without collective (Hartree) effects, for the N1 ADQW
(64/10/29 A, F = 0) and triple well (58/14/34/14/58 A, 55 kV/cm) over sheet densities.
Treatments: bare (single particle), Ando (shifted energies substituted, qwkit sp.chi2_depol = common practice),
TDH (second-order time-dependent Hartree / RPA, qwkit.tdh), LF (macroscopic local-field factors
L(2w) L(w)^2 chi2_bare with L = 1/(1 + chi1_bare,loc/eps_r), chi1 from the bare response at the local density
that reproduces Ando's 1-2 shift). Writes spectra.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, core, tdh

EPS = 11.6; GAM = 0.010
STRUCT = {"ADQW": ([(64, "well"), (10, "barrier"), (29, "well")], 0.0),
          "TW": ([(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")], 55.0)}
NS = [1e11, 3e11, 5e11, 1e12, 2e12, 4e12]


def job(args):
    name, ns = args
    L, F = STRUCT[name]
    r = sp.self_consistent(L, F, ns_cm2=ns, maxit=400)
    E, psi, Z, rho = sp.prepare(r)
    z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
    mk = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
    mb = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc, kernel=False)
    Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=EPS)
    # dominant pump transition for the local-field model
    hw0 = 1.23984 / 10
    i2 = max(range(1, len(E)), key=lambda m: Z[0, m] ** 2 / (1 + ((E[m] - E[0] - hw0) / GAM) ** 2))
    S1 = mk.S[0, i2, 0, i2]; Leff = Z[0, i2] ** 2 * 1e-20 / S1        # m, so that N_loc = ns/Leff gives Ando's shift
    mloc = tdh.TDH(z, psi, E, rho, ns, ns * 1e4 / Leff * 1e-6, gamma=GAM, eps_r=EPS, zc=zc, kernel=False)
    hws = np.linspace(0.09, 0.16, 141)
    out = dict(name=name, ns=ns, E=E.tolist(), rho=rho.tolist(), Z=Z.tolist(), Et=Et.tolist(), i2=int(i2),
               Leff_A=Leff * 1e10, hw=hws.tolist(), bare=[], ando=[], tdh=[], lf=[], a1_tdh=[], a2_tdh=[], a1_ando=[],
               a2_ando=[], a1_bare=[], a2_bare=[])
    w2c = lambda hw: hw * tdh.QE / tdh.HBAR / (3.3 * 2.99792458e8)
    for hw in hws:
        c1k, c2k, _, _ = mk.response(hw); c1kk = mk.chi1(2 * hw)
        c1b, c2b, _, _ = mb.response(hw); c1bb = mb.chi1(2 * hw)
        c2a = -sp.chi2_depol(Et, Z, rho, hw, GAM, N)                          # electron-charge sign convention
        # Ando-substituted absorption (shifted Lorentzians, as in propagation.Medium)
        def a_ando(h):
            s = 0.0
            for l in range(len(E)):
                for m in range(l + 1, len(E)):
                    s += (rho[l] - rho[m]) * (Z[l, m] * 1e-10) ** 2 * GAM / ((Et[m, l] - h) ** 2 + GAM ** 2)
            return h * tdh.QE / tdh.HBAR / (3.3 * 2.99792458e8 * tdh.EPS0) * N * 1e6 * tdh.QE * s
        Lw = 1 / (1 + mloc.chi1(hw) / EPS); L2w = 1 / (1 + mloc.chi1(2 * hw) / EPS)
        c2lf = c2b * Lw ** 2 * L2w
        out["bare"].append([c2b.real, c2b.imag]); out["tdh"].append([c2k.real, c2k.imag])
        out["ando"].append([c2a.real, c2a.imag]); out["lf"].append([c2lf.real, c2lf.imag])
        out["a1_tdh"].append(float(w2c(hw) * c1k.imag)); out["a2_tdh"].append(float(w2c(2 * hw) * c1kk.imag))
        out["a1_bare"].append(float(w2c(hw) * c1b.imag)); out["a2_bare"].append(float(w2c(2 * hw) * c1bb.imag))
        out["a1_ando"].append(float(a_ando(hw))); out["a2_ando"].append(float(a_ando(2 * hw)))
    return out


if __name__ == "__main__":
    with Pool(2) as pool:
        res = pool.map(job, [(n, ns) for n in STRUCT for ns in NS])
    json.dump(res, open(os.path.join(here, "spectra.json"), "w"))
    for o in res:
        b = np.abs(np.array(o["bare"]) @ [1, 1j]); a = np.abs(np.array(o["ando"]) @ [1, 1j])
        t = np.abs(np.array(o["tdh"]) @ [1, 1j]); l = np.abs(np.array(o["lf"]) @ [1, 1j])
        hw = np.array(o["hw"]); E = o["E"]; Et = np.array(o["Et"]); i2 = o["i2"]
        print(f"{o['name']:4s} ns={o['ns']:.0e}: E21={1e3*(E[i2]-E[0]):.1f} shift={1e3*(Et[i2,0]-(E[i2]-E[0])):.1f} meV Leff={o['Leff_A']:.0f}A | "
              f"peak |chi2| (nm/V): bare {b.max()/1e3:.0f}@{hw[b.argmax()]*1e3:.0f} Ando {a.max()/1e3:.0f}@{hw[a.argmax()]*1e3:.0f} "
              f"TDH {t.max()/1e3:.0f}@{hw[t.argmax()]*1e3:.0f} LF {l.max()/1e3:.0f}@{hw[l.argmax()]*1e3:.0f}  TDH/Ando {t.max()/a.max():.2f}", flush=True)
