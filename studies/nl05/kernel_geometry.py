"""NL-05 study 6: geometry of the Hartree kernel. For the ladder (0, m, n) of each survey design at 1e12 cm^-2:
transition charges psi_i psi_j and their cumulative integrals F_ij(z), Ando effective lengths L_ij = z_ij^2 / S_ij,ij,
shape correlation c = S_mn,0m / sqrt(S_mn,mn S_0m,0m), and the identity kappa_exc = c sqrt(L_0m / L_mn).
Writes kernel_geometry.json (profiles for the ADQW and the triple well)."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import sp, tdh
from survey import DESIGNS
S_ = json.load(open(os.path.join(here, "survey.json")))
out = []
for o in S_:
    if "skipped" in o or o["ns"] != 1e12:
        continue
    L, F = DESIGNS[o["name"]]
    r = sp.self_consistent(L, F, ns_cm2=o["ns"], maxit=400); E, psi, Z, rho = sp.prepare(r)
    z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1])
    _, a, b = o["ladder"]
    S = tdh.hartree_kernel(z, psi) * 1e10                         # Angstrom
    dz = z[1] - z[0]
    Zm = (psi.T * (z - zc)) @ psi * dz
    L1 = Zm[0, a] ** 2 / S[0, a, 0, a]; L2 = Zm[a, b] ** 2 / S[a, b, a, b]
    c = S[a, b, 0, a] / np.sqrt(S[a, b, a, b] * S[0, a, 0, a])
    kap = S[a, b, 0, a] / S[0, a, 0, a] * Zm[0, a] / Zm[a, b]
    rec = dict(name=o["name"], L01_A=float(L1), L12_A=float(L2), c=float(c), kappa=float(kap),
               kappa_identity=float(c * np.sqrt(L1 / L2) * np.sign(Zm[0, a] * Zm[a, b])))
    if o["name"].startswith("N1"):
        sel = (z > act[0] - 60) & (z < act[1] + 60)
        Fc = np.cumsum(psi[:, :, None] * psi[:, None, :], axis=0) * dz
        s1 = np.sign(Zm[0, a]); s2 = np.sign(Zm[a, b])
        rec.update(z=(z[sel] - zc).tolist(), q01=(s1 * psi[sel, 0] * psi[sel, a]).tolist(), q12=(s2 * psi[sel, a] * psi[sel, b]).tolist(),
                   F01=(s1 * Fc[sel, 0, a]).tolist(), F12=(s2 * Fc[sel, a, b]).tolist(), z01=float(abs(Zm[0, a])), z12=float(abs(Zm[a, b])))
    out.append(rec)
    print(f"{o['name']:28s} L01={L1:.0f} A L12={L2:.0f} A c={c:+.3f} kappa={kap:+.3f} (c sqrt(L01/L12) = {rec['kappa_identity']:+.3f})")
json.dump(out, open(os.path.join(here, "kernel_geometry.json"), "w"))
