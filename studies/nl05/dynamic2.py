"""NL-05 study 3b: controls for the saturation comparison of dynamic.py (ADQW, 300 K, LO-phonon rates, gamma = 10 meV).
 (a) quasi-static: the rate-equation medium with energies recomputed self-consistently from the saturated populations,
     including the static Hartree potential of the pump-induced charge transfer and the population-dependent Ando
     shifts (no coherent collective dynamics), at the frozen model's photon energy;
 (b) bare control: real-time density matrix without the Hartree kernel vs the bare rate-equation medium, both at the
     photon energy of the TDH run (measures coherent strong-field corrections to the rate model, independent of depolarisation);
 (c) pumped probe: TDH linear response about the time-averaged populations of each real-time run, with the static
     Hartree potential included (all pump-band peaks reported).
Writes dynamic2.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, tdh, scattering, propagation as pr
from dynamic import setup, EPS, GAM, NREF

IS = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

def static_states(z, psi, E, rho_sat, rho0, ns, eps=EPS):
    """States of the static Hamiltonian diag(E) + U_H[rho_sat - rho0] (pump-induced charge transfer between wells),
    with populations rho_sat carried into the new basis. Returns E', psi', populations', Z' (A, about zc)."""
    S = tdh.hartree_kernel(z, psi); K = tdh.QE * ns * 1e4 / (tdh.EPS0 * eps)
    d = np.asarray(rho_sat, float) - np.asarray(rho0, float)
    U = K * np.einsum("ijkk,k->ij", S, d)
    w, C = np.linalg.eigh(np.diag(E) + U)
    p = np.einsum("ki,k,ki->i", C, np.asarray(rho_sat, float), C)
    return w, psi @ C, p


def local_peaks(hws, a):
    a = np.asarray(a)
    return [(float(hws[i] * 1e3), float(a[i] / a.max())) for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] >= a[i + 1] and a[i] > 0.2 * a.max()]

DY = json.load(open(os.path.join(here, "dynamic.json")))


def quasi_static(ns, eps_static=EPS):
    """Rate model whose energies follow the saturated populations: static Hartree potential of the pump-induced
    charge transfer (states rediagonalised) plus Ando depolarisation shifts from the current population differences."""
    r, E, psi, Z, rho, z, zc, N, R, Et = setup(ns)
    hw = next(o["hw_frozen"] for o in DY if o["ns"] == ns)
    dz = z[1] - z[0]
    out = []
    for I_MW in IS:
        I = I_MW * 1e10; rr = rho.copy()
        for it in range(300):
            Es, ps, pq = static_states(z, psi, E, rr, rho, ns, eps=eps_static)
            Zs = (ps.T * (z - zc)) @ ps * dz
            Etq = sp.depol_energies(z, ps, Es, rr, ns, eps_r=EPS)
            med = pr.Medium(dict(E=list(Es), Z=Zs.tolist(), Et=Etq.tolist(), rho=list(rho), N=N), hw, gamma=GAM, nref=NREF, R=R)
            new = med.populations(I, 0.0)
            if np.max(abs(new - rr)) < 1e-7:
                break
            rr = 0.7 * rr + 0.3 * new
        c = med.chi(rr) * 1e12
        out.append(dict(I_MWcm2=I_MW, hw=hw, chi2=[c.real, c.imag], alpha=float(med.alpha(rr)[0]), pops=rr.tolist(),
                        E12_static=float(1e3 * (Es[1] - Es[0])), E12_shifted=float(Etq[1, 0]), iters=it))
    return out


def probe(ns):
    """Linear TDH absorption about the time-averaged populations of each real-time run (a linear-response estimate,
    not part of the real-time solution), with the static Hartree potential of the pump-induced charge transfer
    included; all local maxima of the pump band are reported. 'no_static' omits that potential, for comparison."""
    r, E, psi, Z, rho, z, zc, N, R, Et = setup(ns)
    hws = np.linspace(0.095, 0.15, 221); out = []
    for o in sorted([o for o in DY if o["ns"] == ns], key=lambda o: o["I_MWcm2"]):
        p = np.array(o["pops_tdh"])
        Es, ps, pq = static_states(z, psi, E, p, rho, ns)
        m = tdh.TDH(z, ps, Es, pq, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
        a = [m.chi1(h).imag * h for h in hws]
        m0 = tdh.TDH(z, psi, E, p, ns, N, gamma=GAM, eps_r=EPS, zc=zc)
        a0 = [m0.chi1(h).imag * h for h in hws]
        out.append(dict(I_MWcm2=o["I_MWcm2"], peaks=local_peaks(hws, a), peaks_no_static=local_peaks(hws, a0),
                        E12_static_meV=float(1e3 * (Es[1] - Es[0])), peak_meV=float(hws[int(np.argmax(a))] * 1e3), spectrum=list(map(float, a))))
    return dict(hw=hws.tolist(), runs=out)


def bare_job(args):
    ns, I_MW = args
    r, E, psi, Z, rho, z, zc, N, R, Et = setup(ns)
    mb = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc, R=R, kernel=False)
    hw = next(o["hw_tdh"] for o in DY if o["ns"] == ns and o["I_MWcm2"] == I_MW)   # same photon energy as the TDH run
    I = I_MW * 1e10; E0 = np.sqrt(I / (2 * NREF * tdh.EPS0 * 2.99792458e8))
    Pk, pops = mb.evolve(E0, hw, ramp_ps=1.0)
    c_t = Pk[2] / (tdh.EPS0 * E0 ** 2) * 1e12
    a_t = hw * tdh.QE / tdh.HBAR / (NREF * 2.99792458e8) * (Pk[1] / (tdh.EPS0 * E0)).imag
    Eb = E[:, None] - E[None, :]
    med = pr.Medium(dict(E=list(E), Z=Z.tolist(), Et=Eb.tolist(), rho=list(rho), N=N), hw, gamma=GAM, nref=NREF, R=R)
    rf = med.populations(I, 0.0); c_f = med.chi(rf) * 1e12
    rabi_meV = 2 * E0 * abs(Z[0, 1]) * 1e-10 * 1e3          # 2 q E0 z12 in meV
    return dict(ns=ns, I_MWcm2=I_MW, hw=float(hw), chi2_rt=[c_t.real, c_t.imag], chi2_rate=[float(np.real(c_f)), float(np.imag(c_f))],
                alpha_rt=float(a_t), alpha_rate=float(med.alpha(rf)[0]), pops_rt=list(map(float, pops)), pops_rate=rf.tolist(), rabi_meV=rabi_meV)


if __name__ == "__main__":
    res = dict(quasi_static={str(ns): quasi_static(ns) for ns in [5e11, 2e12]},
               quasi_static_eps13p5={str(ns): quasi_static(ns, 13.5) for ns in [5e11, 2e12]}, probe={str(ns): probe(ns) for ns in [5e11, 2e12]})
    for ns, rows in res["quasi_static"].items():
        fr = sorted([o for o in DY if o["ns"] == float(ns)], key=lambda o: o["I_MWcm2"])
        for q, f in zip(rows, fr):
            print(f"ns={float(ns):.0e} I={q['I_MWcm2']:5.2f}: |chi2| quasi-static {abs(complex(*q['chi2']))/1e3:6.1f} frozen {abs(complex(*f['chi2_frozen']))/1e3:6.1f} "
                  f"TDH {abs(complex(*f['chi2_tdh']))/1e3:6.1f} nm/V; E12 static {q['E12_static']:.1f} E~12 {q['E12_shifted']*1e3:.1f} meV; alpha qs {q['alpha']/100:.0f} 1/cm", flush=True)
    for ns, d in res["probe"].items():
        for r in d["runs"]:
            print(f"probe ns={float(ns):.0e} I={r['I_MWcm2']:5.2f}: E12(static)={r['E12_static_meV']:.1f} peaks {[(round(a,1), round(b,2)) for a,b in r['peaks']]}  without static term {[(round(a,1), round(b,2)) for a,b in r['peaks_no_static']]}", flush=True)
    json.dump(res, open(os.path.join(here, "dynamic2.json"), "w"))
    with Pool(2) as pool:
        res["bare"] = pool.map(bare_job, [(ns, I) for ns in [5e11, 2e12] for I in IS])
    json.dump(res, open(os.path.join(here, "dynamic2.json"), "w"))
    for o in res["bare"]:
        print(f"bare ns={o['ns']:.0e} I={o['I_MWcm2']:5.2f}: 2qE0z12={o['rabi_meV']:.1f} meV |chi2| real-time {abs(complex(*o['chi2_rt']))/1e3:6.1f} rate {abs(complex(*o['chi2_rate']))/1e3:6.1f} "
              f"ratio {abs(complex(*o['chi2_rt']))/abs(complex(*o['chi2_rate'])):.2f}; alpha ratio {o['alpha_rt']/o['alpha_rate']:.2f}", flush=True)
