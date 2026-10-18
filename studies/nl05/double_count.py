"""NL-05 study 5: double counting. An absorption measurement gives the collective (shifted) energies E~.
Using E~ as single-particle energies in chi2 (Ando-equivalent, correct to the extent Ando substitution is) and also
in a macroscopic local-field (full-wave eps_zz) calculation shifts the resonances a second time. For the ADQW and the
triple well: linear peak and |chi2| peak positions and magnitudes for (i) TDH, (ii) Ando substitution, (iii) measured
energies used in both chi2 and the local field (double counting). Writes double_count.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import sp, tdh
from survey import DESIGNS
EPS = 11.6; GAM = 0.010
out = []
for name in ["N1 ADQW 64/10/29", "N1 triple well, 55 kV/cm"]:
    for ns in [1e12, 2e12]:
        L, F = DESIGNS[name]
        r = sp.self_consistent(L, F, ns_cm2=ns, maxit=400); E, psi, Z, rho = sp.prepare(r)
        z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
        mk = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
        Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=EPS)
        Emeas = Et[:, 0].copy()                                   # 'measured' collective energies from the ground state
        # double counting: measured energies as bare energies, local-field (uniform field) applied on top
        mdc = tdh.TDH(z, psi, Emeas, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
        S = mk.S; hw0 = 1.23984 / 10
        i2 = max(range(1, len(E)), key=lambda m: Z[0, m] ** 2 / (1 + ((E[m] - E[0] - hw0) / GAM) ** 2))   # as in spectra.py
        Leff = mk.Z[0, i2] ** 2 * 1e-20 / S[0, i2, 0, i2]
        mloc = tdh.TDH(z, psi, Emeas, rho, ns, ns * 1e4 / Leff * 1e-6, gamma=GAM, eps_r=EPS, zc=zc, kernel=False)
        mb = tdh.TDH(z, psi, Emeas, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc, kernel=False)
        hws = np.linspace(0.095, 0.16, 261)
        lin_t = [mk.chi1(h).imag for h in hws]
        lin_dc = [(mloc.chi1(h) / (1 + mloc.chi1(h) / EPS)).imag for h in hws]
        lin_meas = [mb.chi1(h).imag for h in hws]
        c_t = [abs(mk.chi2(h)) for h in hws]
        c_a = [abs(sp.chi2_depol(Et, Z, rho, h, GAM, N)) for h in hws]
        c_dc = [abs(mb.chi2(h) / (1 + mloc.chi1(h) / EPS) ** 2 / (1 + mloc.chi1(2 * h) / EPS)) for h in hws]
        win = np.abs(hws - Emeas[i2]) < 0.02                       # the pump line (local maxima within +-20 meV)
        pk = lambda y: float(hws[win][np.argmax(np.array(y)[win])])
        o = dict(name=name, ns=ns, i2=i2, E_meas=float(Emeas[i2]), lin_peak_tdh=pk(lin_t), lin_peak_meas=pk(lin_meas),
                 lin_peak_dc=pk(lin_dc), chi2_tdh=[float(max(c_t)), float(hws[np.argmax(c_t)])],
                 chi2_ando=[float(max(c_a)), float(hws[np.argmax(c_a)])], chi2_dc=[float(max(c_dc)), float(hws[np.argmax(c_dc)])])
        out.append(o)
        print(f"{name} ns={ns:.0e}: linear peak TDH {o['lin_peak_tdh']*1e3:.1f} meV, double-counted {o['lin_peak_dc']*1e3:.1f} meV "
              f"(error {1e3*(o['lin_peak_dc']-o['lin_peak_tdh']):+.1f}); |chi2| peak TDH {o['chi2_tdh'][0]/1e3:.0f} nm/V @{o['chi2_tdh'][1]*1e3:.1f}, "
              f"Ando {o['chi2_ando'][0]/1e3:.0f} @{o['chi2_ando'][1]*1e3:.1f}, double-counted {o['chi2_dc'][0]/1e3:.0f} @{o['chi2_dc'][1]*1e3:.1f}")
json.dump(out, open(os.path.join(here, "double_count.json"), "w"), indent=1)
