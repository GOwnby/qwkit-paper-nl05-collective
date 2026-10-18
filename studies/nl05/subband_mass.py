"""NL-05 limitations check: in-plane subband masses from the same Kane nonparabolicity used for the states,
m_i = <psi_i| m_edge(z) [1 + (E_i - V(z))/Eg(z)] |psi_i> (mass at the subband edge), for the ADQW at 1 and
2e12 cm^-2 (300 K self-consistent states). The k-dependence of the pump transition energy is then
E12(k) = E12(0) - (hbar^2 k^2 / 2)(1/m1 - 1/m2), and its spread across the Fermi sea is evaluated at zero
temperature with all electrons in level 1, k_F^2 = 2 pi n_s (spread_E12_meV; Fermi energy pi hbar^2 n_s / m1).
For reference the zero-temperature Fermi energy with the density-of-states mass of the population model
(0.043 m0, the ~110 meV quoted in Sec. 4) is also given.
Writes subband_mass.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import sp
from dynamic import L

HB = 1.054571817e-34; M0 = 9.1093837e-31; QE = 1.602176634e-19

out = []
for ns in [1e12, 2e12]:
    r = sp.self_consistent(L, 0.0, ns_cm2=ns, maxit=400)
    z, V, m, Eg, psi, E = r["z"], r["V"], r["m"], r["Eg"], r["psi"], r["E"]
    dz = z[1] - z[0]
    nrm = np.sum(psi ** 2, axis=0) * dz
    mi = np.array([np.sum(psi[:, i] ** 2 * m * (1 + (E[i] - V) / Eg)) * dz / nrm[i] for i in range(3)])
    kF2 = 2 * np.pi * ns * 1e4                                   # m^-2, spin-degenerate, all in level 1
    spread_a = HB ** 2 * kF2 / 2 * (1 / (mi[0] * M0) - 1 / (mi[1] * M0)) / QE * 1e3
    EF_a = np.pi * HB ** 2 * ns * 1e4 / (mi[0] * M0) / QE * 1e3
    EF_b = np.pi * HB ** 2 * ns * 1e4 / (r["m_dos"] * M0) / QE * 1e3
    out.append(dict(ns=ns, m_i=mi.tolist(), dm21_rel=float(mi[1] / mi[0] - 1), dm31_rel=float(mi[2] / mi[0] - 1),
                    E12_meV=float(1e3 * (E[1] - E[0])), EF0_level1_meV=float(EF_a), EF0_mdos_meV=float(EF_b),
                    spread_E12_meV=float(spread_a)))
json.dump(out, open(os.path.join(here, "subband_mass.json"), "w"), indent=1)
for o in out:
    print(f"ns={o['ns']:.0e}: m1,m2,m3 = " + ", ".join(f"{x:.4f}" for x in o["m_i"]) + f" m0; m2/m1-1 = {100*o['dm21_rel']:.1f}%, "
          f"m3/m1-1 = {100*o['dm31_rel']:.1f}%; E12 = {o['E12_meV']:.1f} meV; zero-T Fermi energy (level 1 only) {o['EF0_level1_meV']:.0f} meV "
          f"(m_dos: {o['EF0_mdos_meV']:.0f} meV); E12 spread across that Fermi sea {o['spread_E12_meV']:.1f} meV")
