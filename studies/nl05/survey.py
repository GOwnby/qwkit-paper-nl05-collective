"""NL-05 study 4: design survey. For several published / N1 active regions at two sheet densities: transition-charge
cross-couplings kappa_exc and kappa_pd, and the peak |chi2| and figure of merit Q = |chi2|^2/(a_w a_2w) (local-field absorption from the same effective medium) and the optimum-length
efficiency merit M = g(u) Q, u = 2 a_w/a_2w (at the |chi2| maximum), and the maximum of M within +-2 gamma of that photon energy from bare, Ando-substituted, macroscopic local-field (LF) and TDH treatments. Writes survey.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, tdh

EPS = 11.6; GAM = 0.010; NREF = 3.3; C = 2.99792458e8
DESIGNS = {"N1 ADQW 64/10/29": ([(64, "well"), (10, "barrier"), (29, "well")], 0.0),
           "N1 triple well, 55 kV/cm": ([(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")], 55.0),
           "Vurgaftman WG(DR) 62/20/29": ([(62, "well"), (20, "barrier"), (29, "well")], 0.0),
           "Vurgaftman SI 26/19.5/57": ([(26, "well"), (19.5, "barrier"), (57, "well")], 0.0),
           "ADQW 50/15/30": ([(50, "well"), (15, "barrier"), (30, "well")], 0.0)}


def job(args):
    name, ns = args
    L, F = DESIGNS[name]
    r = sp.self_consistent(L, F, ns_cm2=ns, maxit=400); E, psi, Z, rho = sp.prepare(r)
    z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
    mk = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
    mb = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc, kernel=False)
    Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=EPS)
    S = mk.S; Zm = mk.Z
    if len(E) < 3:
        return dict(name=name, ns=ns, skipped="fewer than three bound states")
    # ladder: the (m, n) pair with E_n ~ 2 E_m and the largest |z0m zmn zn0| (near-double-resonant SHG triple)
    best = None
    for m_ in range(1, len(E)):
        for n_ in range(1, len(E)):
            if n_ == m_ or E[n_] <= E[m_]:
                continue
            wgt = abs(Zm[0, m_] * Zm[m_, n_] * Zm[n_, 0]) / np.sqrt(1 + (((E[n_] - E[0]) - 2 * (E[m_] - E[0])) / (2 * GAM)) ** 2)
            if best is None or wgt > best[0]:
                best = (wgt, m_, n_)
    _, i2, i3 = best
    kx = S[i2, i3, i2, 0] / S[i2, 0, i2, 0] * Zm[0, i2] / Zm[i2, i3]
    kp = (S[i3, i3, i2, 0] - S[0, 0, i2, 0]) / S[i2, 0, i2, 0] * Zm[0, i2] / (Zm[i3, i3] - Zm[0, 0]) if abs(Zm[i3, i3] - Zm[0, 0]) > 1e-6 else np.nan
    Leff = Zm[0, i2] ** 2 * 1e-20 / S[0, i2, 0, i2]
    Nloc = ns * 1e4 / Leff * 1e-6; fill = N / Nloc
    mloc = tdh.TDH(z, psi, E, rho, ns, Nloc, gamma=GAM, eps_r=EPS, zc=zc, kernel=False)
    hc = 0.5 * (E[i2] - E[0]) + 0.25 * (E[i3] - E[0])
    hws = np.linspace(hc - 0.025, hc + 0.035, 121)
    w2a = lambda h: h * tdh.QE / tdh.HBAR / (NREF * C)

    def a_ando(h):
        s = 0.0
        for l in range(len(E)):
            for m in range(l + 1, len(E)):
                s += (rho[l] - rho[m]) * (Zm[l, m] * 1e-10) ** 2 * GAM / ((Et[m, l] - h) ** 2 + GAM ** 2)
        return w2a(h) / tdh.EPS0 * N * 1e6 * tdh.QE * s
    res = {}
    for key in ["bare", "ando", "lf", "tdh"]:
        best = None; rows = []
        for h in hws:
            if key == "bare":
                c2 = mb.chi2(h); a1 = w2a(h) * mb.chi1(h).imag; a2 = w2a(2 * h) * mb.chi1(2 * h).imag
            elif key == "tdh":
                c2 = mk.chi2(h); a1 = w2a(h) * mk.chi1(h).imag; a2 = w2a(2 * h) * mk.chi1(2 * h).imag
            elif key == "ando":
                c2 = -sp.chi2_depol(Et, Zm, rho, h, GAM, N); a1 = a_ando(h); a2 = a_ando(2 * h)
            else:
                x1 = mloc.chi1(h); x2 = mloc.chi1(2 * h); Lw = 1 / (1 + x1 / EPS); L2 = 1 / (1 + x2 / EPS)
                c2 = mb.chi2(h) * Lw ** 2 * L2
                # effective-medium absorption, consistent with the same local response (period-averaged by N/N_loc)
                a1 = w2a(h) * fill * (x1 * Lw).imag; a2 = w2a(2 * h) * fill * (x2 * L2).imag
            rows.append((h, abs(c2), a1, a2))
            if best is None or abs(c2) > best[0]:
                best = (abs(c2), h, abs(c2) ** 2 / (a1 * a2), a1, a2)
        u = 2 * best[3] / best[4]; g = u ** (-(u + 1) / (u - 1)) if abs(u - 1) > 1e-9 else np.exp(-2.0)
        gf = lambda u: u ** (-(u + 1) / (u - 1)) if abs(u - 1) > 1e-9 else np.exp(-2.0)
        Mw = [(gf(2 * a1 / a2) * c ** 2 / (a1 * a2), h) for h, c, a1, a2 in rows if abs(h - best[1]) <= 2 * GAM + 1e-9]
        Mmax, hM = max(Mw)
        res[key] = dict(chi2_nm=best[0] / 1e3, hw_meV=best[1] * 1e3, Q=best[2], a1=best[3], a2=best[4], u=u, M=g * best[2],
                        M_win=Mmax, hw_M_meV=hM * 1e3)
    return dict(name=name, ns=ns, ladder=[0, i2, i3], E12=float(E[i2] - E[0]), E13=float(E[i3] - E[0]),
                shift12_meV=float((Et[i2, 0] - (E[i2] - E[0])) * 1e3), kappa_exc=float(kx), kappa_pd=float(kp),
                z12=float(Zm[0, i2]), z13=float(Zm[0, i3]), z23=float(Zm[i2, i3]), dmu=float(Zm[i3, i3] - Zm[0, 0]), res=res)


if __name__ == "__main__":
    with Pool(2) as pool:
        out = pool.map(job, [(n, ns) for n in DESIGNS for ns in [1e12, 2e12]])
    json.dump(out, open(os.path.join(here, "survey.json"), "w"), indent=1)
    for o in out:
        if "skipped" in o:
            print(o["name"], o["ns"], o["skipped"]); continue
        r = o["res"]; Qb = r["bare"]["Q"]
        print(f"{o['name']:28s} ns={o['ns']:.0e} E12={o['E12']*1e3:.0f} E13={o['E13']*1e3:.0f} shift={o['shift12_meV']:.1f} "
              f"k_exc={o['kappa_exc']:+.2f} k_pd={o['kappa_pd']:+.2f} | chi2 Ando/TDH/LF/bare = {r['ando']['chi2_nm']:.0f}/{r['tdh']['chi2_nm']:.0f}/{r['lf']['chi2_nm']:.0f}/{r['bare']['chi2_nm']:.0f} "
              f"| Q/Qbare Ando {r['ando']['Q']/Qb:.2f} TDH {r['tdh']['Q']/Qb:.2f} LF {r['lf']['Q']/Qb:.2f} | M/Mbare Ando {r['ando']['M']/r['bare']['M']:.2f} TDH {r['tdh']['M']/r['bare']['M']:.2f} LF {r['lf']['M']/r['bare']['M']:.2f} u_bare {r['bare']['u']:.2f} | M_win/Mbare_win Ando {r['ando']['M_win']/r['bare']['M_win']:.2f} TDH {r['tdh']['M_win']/r['bare']['M_win']:.2f} LF {r['lf']['M_win']/r['bare']['M_win']:.2f} (TDH at {r['tdh']['hw_M_meV']-r['tdh']['hw_meV']:+.0f} meV)", flush=True)
