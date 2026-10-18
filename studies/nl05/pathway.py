"""Closed-form pathway-resolved collective factors vs full second-order TDH, three ladder states of the real ADQW
(pure ground-state population). For drive E (Boyd amplitude), first-order coherences
    x  = rho21(w) = -e E z12 / (E~12 - hw - i g),   y = rho31(w) = -e E z13 / (E~13 - hw - i g)
(E~ = Ando/RPA-shifted energies), the 2w coherence is
    rho31(2w) = -[U32 x + (U33 - U11) y] / (E~13 - 2hw - i g),
    U32 = e E z23 + K (S_23,21 x + S_23,31 y),  U33 - U11 = e E (z33 - z11) + K ((S_33,21 - S_11,21) x + (S_33,31 - S_11,31) y)
so each pathway carries a factor F_p = U_p / (e E z_p). In RWA, F_exc = 1 - kappa_exc Delta12/(E~12 - hw - i g) with
kappa_exc = (S_23,12/S_12,12)(z12/z23), Delta12 = K S_12,12; a uniform depolarisation field (rank-one kernel) gives
kappa = 1 for every pathway. Writes pathway.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import sp, tdh

L = [(64, "well"), (10, "barrier"), (29, "well")]
out = []
for ns in [5e11, 1e12, 2e12, 4e12]:
    r = sp.self_consistent(L, 0.0, ns_cm2=ns, maxit=400); E, psi, Z, rho = sp.prepare(r)
    z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
    p3 = psi[:, :3]; E3 = E[:3]; rho3 = np.array([1.0, 0, 0])
    m = tdh.TDH(z, p3, E3, rho3, ns, N, gamma=0.010, eps_r=11.6, zc=zc)
    S = m.S; K = m.Kpref; Zm = m.Z
    kap_exc = S[1, 2, 1, 0] / S[1, 0, 1, 0] * Zm[0, 1] / Zm[1, 2]
    kap_pd = (S[2, 2, 1, 0] - S[0, 0, 1, 0]) / S[1, 0, 1, 0] * Zm[0, 1] / (Zm[2, 2] - Zm[0, 0])
    D12 = K * S[1, 0, 1, 0]
    rows = []
    for hw in np.linspace(0.11, 0.14, 31):
        c1, c2, rho1, rho2 = m.response(hw)
        x = rho1[1, 0]; y = rho1[2, 0]
        U1 = Zm * 1e-10
        UH = m.UH(rho1)
        U32 = U1[2, 1] + UH[2, 1]; Ud = (U1[2, 2] - U1[0, 0]) + (UH[2, 2] - UH[0, 0])
        # Ando-shifted 1-3 at 2w from the 2w self-consistency of rho31 alone
        E13t = E3[2] - E3[0] + K * S[2, 0, 2, 0]
        r31 = -(U32 * x + Ud * y) / (E13t - 2 * hw - 1j * 0.010)
        P2_path = -tdh.QE * m.N * (Zm[0, 2] * 1e-10) * (r31 + 0) * 1  # z13 * rho31 (one of the two Tr terms)
        P2_full = -tdh.QE * m.N * np.trace(U1 @ rho2)
        Fexc = U32 / U1[2, 1]; Fpd = Ud / (U1[2, 2] - U1[0, 0])
        rows.append(dict(hw=hw, ratio=abs(P2_path / P2_full), Fexc=[Fexc.real, Fexc.imag], Fpd=[Fpd.real, Fpd.imag],
                         Fexc_rwa=[*(lambda f: (f.real, f.imag))(1 - kap_exc * D12 / (E3[1] - E3[0] + D12 - hw - 1j * 0.010))]))
    out.append(dict(ns=ns, kappa_exc=float(kap_exc), kappa_pd=float(kap_pd), D12_meV=float(D12 * 1e3), rows=rows))
    k = np.argmin([abs(rr["hw"] - (E3[1] - E3[0] + D12)) for rr in rows]); rr = rows[k]
    print(f"ns={ns:.0e}: Delta12={D12*1e3:.1f} meV kappa_exc={kap_exc:+.2f} kappa_pd={kap_pd:+.2f} | at hw={rr['hw']*1e3:.0f}: "
          f"F_exc={complex(*rr['Fexc']):.3f} (RWA formula {complex(*rr['Fexc_rwa']):.3f}) F_pd={complex(*rr['Fpd']):.3f}; "
          f"pathway formula / full TDH = {min(r['ratio'] for r in rows):.3f}-{max(r['ratio'] for r in rows):.3f}")
json.dump(out, open(os.path.join(here, "pathway.json"), "w"))
