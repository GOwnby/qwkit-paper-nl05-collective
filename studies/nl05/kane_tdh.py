"""NL-05 check: TDH second-order response with the two-band Kane metric carried into the Hartree (depolarization)
kernel, for the ADQW (64/10/29 A) and the triple well (58/14/34/14/58 A, 55 kV/cm) at 5e11 and 2e12 cm^-2, 300 K.
Following studies/mat01/kane_rpa.py: the transition densities of the kernel are those of the two-component states,
rho_ab = (psi_a psi_b + phi_a phi_b)/(N_a N_b), with phi the valence components (core.kane_valence, interpolated to the
grid) and N_a^2 the two-band norm, F_ab = cumulative (cell-centred) integral of rho_ab, S_abcd = int F_ab F_cd dz;
the dipoles are the two-band dipoles (core.kane_dipoles); no Loewdin step. Energies and populations are unchanged.
The Kane operators are set on qwkit.tdh.TDH after construction (Z and S attributes) and the Liouvillian is rebuilt.
Variants: baseline (Loewdin-orthonormalized one-band envelopes, sp.prepare, as in all other NL-05 studies),
'kane_Z' (two-band dipoles, baseline kernel) and 'kane' (two-band dipoles and kernel).
Reported: peak |chi2| (bare, TDH and energy substitution with the same kernel and dipoles) on the spectra.py photon-energy window, Ando-shifted energies of the pump and
harmonic transitions from each kernel, and the TDH pump-band absorption peaks. Also (ground_check) the peak ground-state
chi(2) with Kane vs Loewdin dipoles for the three ADQWs of the paper in the NL-05 period, as quoted in Sec. 2.
Writes kane_tdh.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, tdh, core as q
from spectra import STRUCT, EPS, GAM

HWS = np.arange(0.090, 0.16001, 0.0001)


def kane_ops(r, nkeep):
    """Two-band dipoles (A, about the active-region centre) and Hartree kernel S (m) for the localized states."""
    z, V, m, Eg, act = r["z"], r["V"], r["m"], r["Eg"], r["active"]
    dz = z[1] - z[0]; zc = 0.5 * (act[0] + act[1])
    psi = r["psi"] / np.sqrt(np.sum(r["psi"] ** 2, axis=0) * dz)
    keep = q.localized(z, psi, act)
    E, P = r["E"][keep][:nkeep], psi[:, keep][:, :nkeep]
    zh, phi = q.kane_valence(z, P, E, V, m, Eg)
    ph = np.zeros_like(P); ph[1:-1] = 0.5 * (phi[1:] + phi[:-1])         # valence components on the grid points
    N = np.sqrt(np.sum(P ** 2, axis=0) * dz + np.sum(phi ** 2, axis=0) * dz)
    rho = (P[:, :, None] * P[:, None, :] + ph[:, :, None] * ph[:, None, :]) / (N[:, None] * N[None, :])[None]
    F = np.cumsum(rho, axis=0) * dz - 0.5 * rho * dz                     # cell-centred, as tdh.hartree_kernel
    n = P.shape[1]; F2 = F.reshape(F.shape[0], n * n)
    S = (F2.T @ F2).reshape(n, n, n, n) * dz * 1e-10
    Z, ov = q.kane_dipoles(z, P, E, V, m, Eg, zc)
    return E, Z, S, ov


def local_peaks(hws, a):
    a = np.asarray(a)
    return [float(hws[i] * 1e3) for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] >= a[i + 1] and a[i] > 0.2 * a.max()]


def ando_matrix(m, E, rho):
    """Signed Ando-shifted transition-energy matrix (eV) from the kernel of model m (as sp.depol_energies)."""
    n = len(E); Et = E[:, None] - E[None, :]
    for i in range(n):
        for j in range(i + 1, n):
            Eij = E[j] - E[i]; D = m.Kpref * (rho[i] - rho[j]) * m.S[i, j, i, j]
            if Eij > 0 and D != 0:
                Et[j, i] = Eij * np.sqrt(max(1 + 2 * D / Eij, 0.0)); Et[i, j] = -Et[j, i]
    return Et


def ando(m, E, rho, i, j):
    """Ando-shifted energy (meV) of transition i -> j from the kernel and populations of model m."""
    Eij = E[j] - E[i]; D = m.Kpref * (rho[i] - rho[j]) * m.S[i, j, i, j]
    return 1e3 * np.sqrt(Eij ** 2 + 2 * Eij * D), 1e3 * D


def job(args):
    name, ns = args
    L, F = STRUCT[name]
    r = sp.self_consistent(L, F, ns_cm2=ns, maxit=400)
    E, psi, Z, rho = sp.prepare(r)
    z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
    EK, ZK, SK, ov = kane_ops(r, len(E))
    assert np.allclose(EK, E)
    hw0 = 1.23984 / 10                                                    # pump transition as in spectra.py
    i2 = max(range(1, len(E)), key=lambda m_: Z[0, m_] ** 2 / (1 + ((E[m_] - E[0] - hw0) / GAM) ** 2))
    i3 = max(range(i2 + 1, len(E)), key=lambda n_: abs(Z[0, i2] * Z[i2, n_] * Z[n_, 0]) / np.sqrt(1 + (((E[n_] - E[0]) - 2 * (E[i2] - E[0])) / (2 * GAM)) ** 2))
    res = dict(name=name, ns=ns, ladder=[0, int(i2), int(i3)], two_band_overlap=ov,
               z_pump=[float(Z[0, i2]), float(ZK[0, i2])], z_up=[float(Z[i2, i3]), float(ZK[i2, i3])], z_harm=[float(Z[0, i3]), float(ZK[0, i3])])
    for tag in ("baseline", "kane_Z", "kane"):
        out = {}
        for kern in (False, True):
            m = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc, kernel=kern)
            if tag in ("kane_Z", "kane"):
                m.Z = ZK
            if tag == "kane":
                m.S = SK
            m._build()                                                    # Liouvillian with the overridden kernel
            c2 = []; a1 = []
            for h in HWS:
                c1, cc2, _, _ = m.response(h); c2.append(abs(cc2)); a1.append(c1.imag * h)
            k = "tdh" if kern else "bare"
            ip = int(np.argmax(c2))
            out[k + "_chi2_nm"] = float(c2[ip] / 1e3); out[k + "_chi2_hw_meV"] = float(HWS[ip] * 1e3)
            if kern:
                out["abs_peaks_meV"] = local_peaks(HWS, a1)
                out["Et_pump_meV"], out["D_pump_meV"] = ando(m, E, rho, 0, i2)
                out["Et_harm_meV"], out["D_harm_meV"] = ando(m, E, rho, 0, i3)
                out["Et_up_meV"], out["D_up_meV"] = ando(m, E, rho, i2, i3)
                Et = ando_matrix(m, E, rho)                                # energy substitution with this kernel and Z
                ca = [abs(sp.chi2_depol(Et, m.Z, rho, h, GAM, N)) for h in HWS]
                out["ando_chi2_nm"] = float(max(ca) / 1e3); out["tdh_over_ando"] = out["tdh_chi2_nm"] / out["ando_chi2_nm"]
        res[tag] = out
    return res


ADQWS = {"64/10/29": [(64, "well"), (10, "barrier"), (29, "well")], "62/20/29": [(62, "well"), (20, "barrier"), (29, "well")],
         "50/15/30": [(50, "well"), (15, "barrier"), (30, "well")]}


def ground_check(name):
    """Peak ground-state chi(2) (core.chi2_shg, gamma = 10 meV, nominal N) with two-band Kane dipoles vs Loewdin
    dipoles, as studies/kane/kane_check_all.py but in the NL-05 period (sp.self_consistent defaults: 100 A spacers on
    each side), 5e11 cm^-2, 300 K states, four lowest localized states (or all, if fewer)."""
    r = sp.self_consistent(ADQWS[name], 0.0, ns_cm2=5e11, maxit=400)
    E, psi, ZL, rho = sp.prepare(r, nkeep=4)
    EK, ZK, SK, ov = kane_ops(r, len(E))
    hw = np.arange(0.5, 1.5, 0.002) * (E[1] - E[0])
    cL = max(abs(q.chi2_shg(E, ZL, h, 0.010, 1e18)) for h in hw)
    cK = max(abs(q.chi2_shg(EK, ZK, h, 0.010, 1e18)) for h in hw)
    return dict(name=name, n_localized=len(E), two_band_overlap=ov, z13=[float(ZL[0, 2]), float(ZK[0, 2])], chi2_ratio=float(cK / cL))


if __name__ == "__main__":
    with Pool(2) as pool:
        res = pool.map(job, [(n, ns) for n in STRUCT for ns in (5e11, 2e12)])
        gc = pool.map(ground_check, list(ADQWS))
    json.dump(dict(tdh=res, ground=gc), open(os.path.join(here, "kane_tdh.json"), "w"), indent=1)
    for g in gc:
        print(f"ground-state check ADQW {g['name']}: {g['n_localized']} localized states, two-band overlap {g['two_band_overlap']:.1e}, "
              f"z13 {g['z13'][0]:.2f}/{g['z13'][1]:.2f} A, peak chi2 Kane/Loewdin {g['chi2_ratio']:.4f} ({g['chi2_ratio']-1:+.2%})", flush=True)
    for o in res:
        b, kz, k = o["baseline"], o["kane_Z"], o["kane"]
        print(f"{o['name']:4s} ns={o['ns']:.0e} ladder {[i + 1 for i in o['ladder']]} two-band overlap {o['two_band_overlap']:.1e}; "
              f"z_pump {o['z_pump'][0]:.2f}/{o['z_pump'][1]:.2f} z_up {o['z_up'][0]:.2f}/{o['z_up'][1]:.2f} z_harm {o['z_harm'][0]:.2f}/{o['z_harm'][1]:.2f} A")
        for tag, d in (("baseline", b), ("kane_Z", kz), ("kane", k)):
            print(f"   {tag:8s}: |chi2| bare {d['bare_chi2_nm']:.2f} TDH {d['tdh_chi2_nm']:.2f} nm/V @{d['tdh_chi2_hw_meV']:.1f}; "
                  f"Ando pump {d['Et_pump_meV']:.2f} (shift {d['D_pump_meV']:.3f}) harm {d['Et_harm_meV']:.2f} (shift {d['D_harm_meV']:.3f}) "
                  f"upper {d['Et_up_meV']:.2f} meV; Ando |chi2| {d['ando_chi2_nm']:.2f} TDH/Ando {d['tdh_over_ando']:.3f}; TDH absorption peaks {[round(x, 1) for x in d['abs_peaks_meV']]}")
        print(f"   Kane vs baseline: TDH |chi2| {k['tdh_chi2_nm']/b['tdh_chi2_nm']-1:+.2%} (bare {k['bare_chi2_nm']/b['bare_chi2_nm']-1:+.2%}); "
              f"kernel alone (kane vs kane_Z): TDH |chi2| {k['tdh_chi2_nm']/kz['tdh_chi2_nm']-1:+.2%}, pump shift {k['D_pump_meV']/b['D_pump_meV']-1:+.2%}, "
              f"E~pump {k['Et_pump_meV']-b['Et_pump_meV']:+.3f} meV, E~harm {k['Et_harm_meV']-b['Et_harm_meV']:+.3f} meV, "
              f"TDH/bare {k['tdh_chi2_nm']/k['bare_chi2_nm']:.3f} vs {b['tdh_chi2_nm']/b['bare_chi2_nm']:.3f}", flush=True)
