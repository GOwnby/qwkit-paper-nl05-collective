"""NL-05 study 7: exact decomposition of the second-order TDH polarisation of the three-level ADQW (ground state only)
by the matrix elements of the first-order effective potential U~ = e E z + U_H[rho1] that drive the 2w coherences:
    (-2iw - L0) rho2 = -(i/hbar)[U~, rho1]  is linear in U~, so P2 = sum over element groups G and over (bare, Hartree):
    G = 'exc' (U~_23, U~_32), 'diag' (U~_ii), 'rest' (U~_12, U~_13 and transposes).
Also the pathway factors F_exc = U~_32/(eE z23), F_d2 = (U~_22 - U~_11)/(eE (z22 - z11)), and their geometric couplings
kappa_exc, kappa_d2 = (S_22,12 - S_11,12)/S_12,12 * z12/(z22 - z11). Writes decompose.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from qwkit import sp, tdh
L = [(64, "well"), (10, "barrier"), (29, "well")]; GAM = 0.010; EPS = 11.6
out = []
for ns in [5e11, 1e12, 2e12, 4e12]:
    r = sp.self_consistent(L, 0.0, ns_cm2=ns, maxit=400); E, psi, Z, rho = sp.prepare(r)
    z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
    rr = np.array([1.0, 0, 0])
    m = tdh.TDH(z, psi[:, :3], E[:3], rr, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
    Et = sp.depol_energies(z, psi[:, :3], E[:3], rr, ns, eps_r=EPS)
    S = m.S; K = m.Kpref; Zm = m.Z
    hws = np.linspace(0.105, 0.145, 161)
    t = np.array([abs(m.chi2(h)) for h in hws]); a = np.array([abs(sp.chi2_depol(Et, Zm, rr, h, GAM, N)) for h in hws])
    h = hws[t.argmax()]; w = h / tdh.HBAR_EVS
    c1, c2, rho1, rho2 = m.response(h)
    U1 = Zm * 1e-10; UH = m.UH(rho1)
    masks = {"exc": np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], bool), "diag": np.eye(3, dtype=bool)}
    masks["rest"] = ~(masks["exc"] | masks["diag"])
    P = lambda Ut: -tdh.QE * m.N * np.trace(U1 @ m._solve(2 * w, -(1j / tdh.HBAR_EVS) * (Ut @ rho1 - rho1 @ Ut))) / tdh.EPS0 * 1e12
    parts = {}
    for g, mk in masks.items():
        parts[g + "_bare"] = P(np.where(mk, U1, 0)); parts[g + "_H"] = P(np.where(mk, UH, 0))
    tot = sum(parts.values())
    D12 = K * S[1, 0, 1, 0]
    kx = S[1, 2, 1, 0] / S[1, 0, 1, 0] * Zm[0, 1] / Zm[1, 2]
    kd2 = (S[1, 1, 1, 0] - S[0, 0, 1, 0]) / S[1, 0, 1, 0] * Zm[0, 1] / (Zm[1, 1] - Zm[0, 0])
    Fx = (U1[2, 1] + UH[2, 1]) / U1[2, 1]; Fd = (U1[1, 1] - U1[0, 0] + UH[1, 1] - UH[0, 0]) / (U1[1, 1] - U1[0, 0])
    o = dict(ns=ns, hw=h, D12_meV=D12 * 1e3, kappa_exc=kx, kappa_d2=kd2, dmu12_A=Zm[1, 1] - Zm[0, 0], z12=Zm[0, 1], z23=Zm[1, 2],
             z13=Zm[0, 2], Fexc=[Fx.real, Fx.imag], Fd2=[Fd.real, Fd.imag], tdh_over_ando=t.max() / a.max(),
             parts={k: [v.real / abs(tot), v.imag / abs(tot)] for k, v in parts.items()}, check=abs(tot - c2) / abs(c2))
    out.append(o)
    print(f"ns={ns:.0e} hw={h*1e3:.1f} D12={D12*1e3:.1f} meV dmu12={o['dmu12_A']:+.1f} A k_exc={kx:.2f} k_d2={kd2:.2f} |Fexc|={abs(Fx):.2f} |Fd2|={abs(Fd):.2f} "
          f"TDH/Ando={o['tdh_over_ando']:.3f} check={o['check']:.1e}")
    for k, v in parts.items():
        print(f"    {k:10s} {v/abs(tot):.3f}")


# ---- part 2: the two roles of the (12)x(23) kernel block, with the self-consistent thermal populations at 77 K and 300 K
def cut(S):
    S = S.copy()
    for p in [(0, 1), (1, 0)]:
        for q in [(1, 2), (2, 1)]:
            S[p[0], p[1], q[0], q[1]] = 0; S[q[0], q[1], p[0], p[1]] = 0
    return S


roles = []
for T in [77.0, 300.0]:
    for ns in [1e12, 2e12]:
        r = sp.self_consistent(L, 0.0, ns_cm2=ns, T=T, maxit=400); E, psi, Z, rho = sp.prepare(r)
        z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
        Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=EPS)
        hws = np.linspace(0.105, 0.145, 161)
        a = np.array([abs(sp.chi2_depol(Et, Z, rho, h, GAM, N)) for h in hws]).max()
        res = {}
        for mode in ["full", "no_source", "no_propagation", "none"]:
            m = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
            Sf = m.S.copy(); Sc = cut(Sf)
            m.S = Sc if mode in ("no_propagation", "none") else Sf; m._build()
            m.S = Sc if mode in ("no_source", "none") else Sf
            res[mode] = float(max(abs(m.chi2(h)) for h in hws) / a)
        roles.append(dict(T=T, ns=ns, rho=rho[:3].tolist(), E12=float(E[1] - E[0]), E23=float(E[2] - E[1]), ando_nm=a / 1e3, ratio=res))
        print(f"T={T:.0f} K ns={ns:.0e} rho={np.round(rho[:3], 3)} E12={1e3*(E[1]-E[0]):.0f} E23={1e3*(E[2]-E[1]):.0f} meV  TDH/Ando: "
              + " ".join(f"{k}={v:.3f}" for k, v in res.items()))
json.dump(dict(decomposition=out, roles=roles), open(os.path.join(here, "decompose.json"), "w"), indent=1)
