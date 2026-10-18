"""NL-05 study 3c: population brackets of the 1-2-3 ladder term of chi(2) under saturation (ADQW, 300 K).
For the frozen-shift rate model (equilibrium Ando energies, saturated rate-equation populations from dynamic.json)
and the quasi-static rate model (states rediagonalised in the static Hartree potential of the saturated populations,
Ando shifts recomputed; populations from dynamic2.json), both at the frozen model's photon energy, evaluates
    A = (rho1 - rho2)/(E~12 - hw - i gamma),  B = (rho2 - rho3)/(E~23 - hw - i gamma),  |A - B|/|A|,
the ratio quasi-static/frozen of |A - B| (the bracket ratio, compared with the |chi2| ratio), and the
quasi-static energies E12, E~12, E~13 and the single-particle E23. No new dynamics; deterministic post-processing.
Writes brackets.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import sp
from dynamic import setup, EPS, GAM
from dynamic2 import static_states

DY = json.load(open(os.path.join(here, "dynamic.json")))
D2 = json.load(open(os.path.join(here, "dynamic2.json")))


def bracket(Et, rho, hw):
    A = (rho[0] - rho[1]) / (Et[1, 0] - hw - 1j * GAM)
    B = (rho[1] - rho[2]) / (Et[2, 1] - hw - 1j * GAM)
    return A, B


out = {}
for ns in [5e11, 2e12]:
    r, E, psi, Z, rho, z, zc, N, R, Et = setup(ns)
    fr = sorted([o for o in DY if o["ns"] == ns], key=lambda o: o["I_MWcm2"])
    rows = []
    for q, f in zip(D2["quasi_static"][str(ns)], fr):
        assert q["I_MWcm2"] == f["I_MWcm2"]
        hw = q["hw"]; rr = np.array(q["pops"])
        Es, ps, pq = static_states(z, psi, E, rr, rho, ns)
        Etq = sp.depol_energies(z, ps, Es, rr, ns, eps_r=EPS)
        Af, Bf = bracket(Et, np.array(f["pops_frozen"]), hw)
        Aq, Bq = bracket(Etq, rr, hw)
        rows.append(dict(I_MWcm2=q["I_MWcm2"], hw_meV=1e3 * hw,
                         frozen_AmB_over_A=abs(Af - Bf) / abs(Af), qs_AmB_over_A=abs(Aq - Bq) / abs(Aq),
                         bracket_ratio=abs(Aq - Bq) / abs(Af - Bf),
                         chi2_ratio_qs_frozen=abs(complex(*q["chi2"])) / abs(complex(*f["chi2_frozen"])),
                         E12_meV=1e3 * (Es[1] - Es[0]), E23_meV=1e3 * (Es[2] - Es[1]),
                         Et12_meV=1e3 * Etq[1, 0], Et13_meV=1e3 * Etq[2, 0], Et23_meV=1e3 * Etq[2, 1]))
    out[str(ns)] = dict(equilibrium=dict(E12_meV=1e3 * (E[1] - E[0]), E23_meV=1e3 * (E[2] - E[1]), Et12_meV=1e3 * Et[1, 0],
                                         Et13_meV=1e3 * Et[2, 0], Et23_meV=1e3 * Et[2, 1]), rows=rows)

json.dump(out, open(os.path.join(here, "brackets.json"), "w"), indent=1)
for ns, d in out.items():
    e = d["equilibrium"]
    print(f"ns={float(ns):.0e} equilibrium: E12 {e['E12_meV']:.1f} E23 {e['E23_meV']:.1f} E~12 {e['Et12_meV']:.1f} E~13 {e['Et13_meV']:.1f} E~23 {e['Et23_meV']:.1f} meV")
    for o in d["rows"]:
        print(f"  I={o['I_MWcm2']:5.2f} hw={o['hw_meV']:.1f}: |A-B|/|A| frozen {o['frozen_AmB_over_A']:.3f} quasi-static {o['qs_AmB_over_A']:.3f}; "
              f"bracket ratio {o['bracket_ratio']:.3f} (|chi2| ratio {o['chi2_ratio_qs_frozen']:.3f}); quasi-static E12 {o['E12_meV']:.1f} "
              f"E23 {o['E23_meV']:.1f} E~12 {o['Et12_meV']:.1f} E~13 {o['Et13_meV']:.1f} E~23 {o['Et23_meV']:.1f}")
