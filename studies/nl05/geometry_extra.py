"""NL-05 supplementary geometry numbers (300 K self-consistent states, as in survey.py):
 (a) effective length of the harmonic transition, L_0n = z_0n^2 / S_0n,0n, relative to that of the pump
     transition L_0m, for the survey ladder (0, m, n) of every design at 1 and 2e12 cm^-2;
 (b) triple well under bias: energy of level 4 above the potential at the downhill end of the period, energy of
     the fifth state above level 2, and the single-particle and Ando-shifted 2-3 transition energies.
Writes geometry_extra.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import sp, tdh
from survey import DESIGNS, EPS

SV = json.load(open(os.path.join(here, "survey.json")))
out = dict(harmonic=[], tw=[])
for o in SV:
    if "skipped" in o:
        continue
    L, F = DESIGNS[o["name"]]
    r = sp.self_consistent(L, F, ns_cm2=o["ns"], maxit=400); E, psi, Z, rho = sp.prepare(r)
    z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); dz = z[1] - z[0]
    _, a, b = o["ladder"]
    S = tdh.hartree_kernel(z, psi) * 1e10                         # Angstrom
    Zm = (psi.T * (z - zc)) @ psi * dz
    Lp = Zm[0, a] ** 2 / S[0, a, 0, a]; Lh = Zm[0, b] ** 2 / S[0, b, 0, b]
    out["harmonic"].append(dict(name=o["name"], ns=o["ns"], L_pump_A=float(Lp), L_harm_A=float(Lh), ratio=float(Lh / Lp)))
    print(f"{o['name']:28s} ns={o['ns']:.0e} ladder 1-{a+1}-{b+1}: L_pump {Lp:.0f} A, L_harm {Lh:.0f} A, L_harm/L_pump {Lh/Lp:.2f}", flush=True)

L, F = DESIGNS["N1 triple well, 55 kV/cm"]
for ns in [1e12, 2e12, 4e12]:
    r = sp.self_consistent(L, F, ns_cm2=ns, maxit=400); E, psi, Z, rho = sp.prepare(r)
    V = r["V"]; Vdown = min(V[0], V[-1])
    Et = sp.depol_energies(r["z"], psi, E, rho, ns, eps_r=EPS)
    rec = dict(ns=ns, E_meV=(1e3 * (E - E[0])).tolist(), E4_above_downhill_meV=float(1e3 * (E[3] - Vdown)),
               E5_minus_E2_meV=float(1e3 * (E[4] - E[1])) if len(E) > 4 else None,
               E23_meV=float(1e3 * (E[2] - E[1])), Et23_meV=float(1e3 * Et[2, 1]), rho=rho.tolist())
    out["tw"].append(rec)
    print(f"TW ns={ns:.0e}: E-E1 = {np.round(rec['E_meV'], 1)} meV; level 4 - V(downhill end) = {rec['E4_above_downhill_meV']:.1f} meV; "
          f"E5 - E2 = {rec['E5_minus_E2_meV']:.1f} meV; E23 = {rec['E23_meV']:.1f}, E~23 = {rec['Et23_meV']:.1f} meV; rho2 = {rho[1]:.3f}", flush=True)
json.dump(out, open(os.path.join(here, "geometry_extra.json"), "w"), indent=1)
