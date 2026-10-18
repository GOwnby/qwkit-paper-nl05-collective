"""NL-05 study 8: which kernel blocks raise or lower |chi2|, the level-2 counterfactual, and the TW pathways.
Kernel blocks (pairs ij, kl): 'self' = (ij,ij)/(ij,ji) with i != j (Ando shifts); 'cross' = two different off-diagonal
pairs (e.g. (12),(23)); 'diag' = any element with a diagonal pair (ii, kl). For each case, peak |chi2| relative to energy
substitution (Ando) with kernels: full, no diag, self only (kappa = 0 limit of Eq. 6), self + cross, self + diag.
Cases: ADQW (states 1-3) and TW ladder (states 1, 3, 4) with only the ground state populated, at 1 and 2e12 cm^-2;
ADQW and TW with all bound states and self-consistent 300 K populations. Counterfactual: ADQW at 2e12 with the 300 K
states but rho2 -> 0 (all electrons in level 1): chi2 ratio and linear absorption peaks.
Also kappa_d (and kappa_exc) of the TW ladder, and the Ando-vs-TDH kernel discretisation difference. Writes masks.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import sp, tdh
EPS = 11.6; GAM = 0.010
ST = {"ADQW": ([(64, "well"), (10, "barrier"), (29, "well")], 0.0, [0, 1, 2]),
      "TW": ([(58, "well"), (14, "barrier"), (34, "well"), (14, "barrier"), (58, "well")], 55.0, [0, 2, 3])}


def blockmask(n, kind):
    M = np.zeros((n, n, n, n), bool)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    d1 = i == j; d2 = k == l
                    if d1 or d2:
                        M[i, j, k, l] = kind == "diag"
                    elif {i, j} == {k, l}:
                        M[i, j, k, l] = kind == "self"
                    else:
                        M[i, j, k, l] = kind == "cross"
    return M


def peak_ratio(z, psi, E, rho, ns, N, zc, keep, hws):
    m = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
    Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=EPS)
    a = max(abs(sp.chi2_depol(Et, m.Z, rho, h, GAM, N)) for h in hws)
    n = len(E)
    if keep is not None:
        mask = np.zeros((n, n, n, n), bool)
        for k in keep:
            mask |= blockmask(n, k)
        m.S = np.where(mask, m.S, 0.0); m._build()
    return max(abs(m.chi2(h)) for h in hws) / a



if __name__ == "__main__":
    out = dict(masks=[], counterfactual=None, tw_kappa=[], discretisation=[])
    CASES = {"full": None, "no_diag": ["self", "cross"], "self_only": ["self"], "self+cross": ["self", "cross"][:2],
             "self+diag": ["self", "diag"]}
    CASES["self+cross"] = ["self", "cross"]
    for name, (L, F, lad) in ST.items():
        for ns in [1e12, 2e12]:
            r = sp.self_consistent(L, F, ns_cm2=ns, maxit=400); E, psi, Z, rho = sp.prepare(r)
            z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
            hc = 0.5 * (E[lad[1]] - E[0]) + 0.25 * (E[lad[2]] - E[0]); hws = np.linspace(hc - 0.025, hc + 0.035, 121)
            for pop in ["ground3", "thermal"]:
                if pop == "ground3":
                    Ps, Es, Rs = psi[:, lad], E[lad], np.array([1.0, 0, 0])
                else:
                    Ps, Es, Rs = psi, E, rho
                res = {k: float(peak_ratio(z, Ps, Es, Rs, ns, N, zc, v, hws)) for k, v in CASES.items()}
                rec = dict(name=name, ns=ns, pop=pop, rho=list(map(float, Rs[:4])), ratio=res)
                out["masks"].append(rec)
                print(f"{name} {ns:.0e} {pop:8s} " + " ".join(f"{k}={v:.3f}" for k, v in res.items()), flush=True)
            if name == "TW":
                S = tdh.hartree_kernel(z, psi); Zc = (psi.T * (z - zc)) @ psi * (z[1] - z[0])
                a, b = lad[1], lad[2]
                kx = S[a, b, a, 0] / S[a, 0, a, 0] * Zc[0, a] / Zc[a, b]
                kd = (S[a, a, a, 0] - S[0, 0, a, 0]) / S[a, 0, a, 0] * Zc[0, a] / (Zc[a, a] - Zc[0, 0])
                out["tw_kappa"].append(dict(ns=ns, kappa_exc=float(kx), kappa_d=float(kd), dmu_nm=float((Zc[a, a] - Zc[0, 0]) / 10),
                                            E=[float(e) for e in E[:5]], rho=[float(x) for x in rho[:5]]))
                print(f"   TW ladder kappa_exc={kx:.2f} kappa_d={kd:.2f} z33-z11={(Zc[a, a]-Zc[0, 0])/10:+.2f} nm; E-E1 = "
                      + str(np.round(1e3 * (E[:5] - E[0]), 1)) + " meV; rho = " + str(np.round(rho[:5], 3)), flush=True)
            # discretisation: Ando shift from sp (uncentred) vs TDH kernel (cell-centred)
            S = tdh.hartree_kernel(z, psi); K = tdh.QE * ns * 1e4 / (tdh.EPS0 * EPS)
            i2 = lad[1]; Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=EPS)
            E12 = E[i2] - E[0]; Et_c = E12 * np.sqrt(1 + 2 * K * (rho[0] - rho[i2]) * S[0, i2, 0, i2] / E12)
            out["discretisation"].append(dict(name=name, ns=ns, shift_sp_meV=float(1e3 * (Et[i2, 0] - E12)), shift_centred_meV=float(1e3 * (Et_c - E12))))
            print(f"   shift sp {1e3*(Et[i2,0]-E12):.3f} meV vs cell-centred {1e3*(Et_c-E12):.3f} meV", flush=True)

    # counterfactual: ADQW 2e12, 300 K states, populations thermal vs rho2 -> 0
    L, F, lad = ST["ADQW"]; ns = 2e12
    r = sp.self_consistent(L, F, ns_cm2=ns, maxit=400); E, psi, Z, rho = sp.prepare(r)
    z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
    hws = np.linspace(0.095, 0.155, 241); cf = {}
    for lab, R in [("thermal", rho), ("rho2_zero", np.array([1.0] + [0.0] * (len(E) - 1)))]:
        m = tdh.TDH(z, psi, E, R, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
        Et = sp.depol_energies(z, psi, E, R, ns, eps_r=EPS)
        t = np.array([abs(m.chi2(h)) for h in hws]); a = np.array([abs(sp.chi2_depol(Et, m.Z, R, h, GAM, N)) for h in hws])
        ab = np.array([h * m.chi1(h).imag for h in hws])
        pk = [float(hws[i] * 1e3) for i in range(1, len(hws) - 1) if ab[i] > ab[i - 1] and ab[i] > ab[i + 1] and ab[i] > 0.2 * ab.max()]
        cf[lab] = dict(rho=list(map(float, R)), tdh_over_ando=float(t.max() / a.max()), tdh_nm=float(t.max() / 1e3), ando_nm=float(a.max() / 1e3),
                       abs_peaks_meV=pk, E12t=float(Et[1, 0] * 1e3), E23t=float(Et[2, 1] * 1e3) if R[1] != R[2] else None,
                       E12=float(1e3 * (E[1] - E[0])), E23=float(1e3 * (E[2] - E[1])))
        print("counterfactual", lab, cf[lab], flush=True)
    out["counterfactual"] = cf
    json.dump(out, open(os.path.join(here, "masks.json"), "w"), indent=1)
