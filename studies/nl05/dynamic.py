"""NL-05 study 3: saturation with dynamic depolarisation. Real-time TDH (qwkit.tdh.evolve, LO-phonon rate matrix,
gamma = 10 meV) vs the frozen-shift rate-equation medium used in NL-02 (propagation.Medium with Ando-shifted
energies fixed at their equilibrium values), for the ADQW at two sheet densities, at the photon energy of each
model's small-signal |chi2| maximum. Outputs chi2_eff(I) = P2/E^2 and alpha(I) (drive-referenced). Writes dynamic.json."""
import os, sys, json
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(here, "..", ".."))
import numpy as np
from multiprocessing import Pool
from qwkit import sp, tdh, scattering, propagation as pr

EPS = 11.6; GAM = 0.010; NREF = 3.3
L = [(64, "well"), (10, "barrier"), (29, "well")]


def setup(ns):
    r = sp.self_consistent(L, 0.0, ns_cm2=ns, maxit=400); E, psi, Z, rho = sp.prepare(r)
    z = r["z"]; act = r["active"]; zc = 0.5 * (act[0] + act[1]); N = ns / ((act[1] - act[0] + 200) * 1e-8)
    R = scattering.rate_matrix(z - zc, psi, E)
    Et = sp.depol_energies(z, psi, E, rho, ns, eps_r=EPS)
    return r, E, psi, Z, rho, z, zc, N, R, Et


def job(args):
    ns, I_MW = args
    r, E, psi, Z, rho, z, zc, N, R, Et = setup(ns)
    m = tdh.TDH(z, psi, E, rho, ns, N, gamma=GAM, eps_r=EPS, zc=zc, R=R)
    st = dict(E=list(E), Z=Z.tolist(), Et=Et.tolist(), rho=list(rho), N=N)
    med = pr.Medium(st, 0.124, gamma=GAM, nref=NREF, R=R)
    hws = np.linspace(0.115, 0.13, 31)
    hw_t = hws[np.argmax([abs(m.chi2(h)) for h in hws])]
    I = I_MW * 1e10
    E0 = np.sqrt(I / (2 * NREF * tdh.EPS0 * 2.99792458e8))           # Boyd amplitude
    Pk, pops = m.evolve(E0, hw_t, ramp_ps=1.0)
    chi2_t = Pk[2] / (tdh.EPS0 * E0 ** 2) * 1e12
    chi1_t = Pk[1] / (tdh.EPS0 * E0)
    a_t = hw_t * tdh.QE / tdh.HBAR / (NREF * 2.99792458e8) * chi1_t.imag
    # frozen-shift rate model at its own optimum photon energy
    meds = {h: pr.Medium(st, h, gamma=GAM, nref=NREF, R=R) for h in hws}
    hw_f = max(hws, key=lambda h: abs(meds[h].chi(meds[h].rho0)))
    mf = meds[hw_f]; rf = mf.populations(I, 0.0)
    chi2_f = mf.chi(rf) * 1e12; a_f = mf.alpha(rf)[0]
    return dict(ns=ns, I_MWcm2=I_MW, hw_tdh=float(hw_t), hw_frozen=float(hw_f), chi2_tdh=[chi2_t.real, chi2_t.imag],
                alpha_tdh=float(a_t), pops_tdh=list(map(float, pops)), chi2_frozen=[float(np.real(chi2_f)), float(np.imag(chi2_f))],
                alpha_frozen=float(a_f), pops_frozen=list(map(float, rf)))


if __name__ == "__main__":
    Is = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    with Pool(2) as pool:
        res = pool.map(job, [(ns, I) for ns in [5e11, 2e12] for I in Is])
    json.dump(res, open(os.path.join(here, "dynamic.json"), "w"), indent=1)
    for o in res:
        ct = abs(complex(*o["chi2_tdh"])); cf = abs(complex(*o["chi2_frozen"]))
        print(f"ns={o['ns']:.0e} I={o['I_MWcm2']:6.2f} MW/cm2: |chi2| TDH {ct/1e3:7.1f} frozen {cf/1e3:7.1f} nm/V  "
              f"alpha TDH {o['alpha_tdh']/100:8.0f} frozen {o['alpha_frozen']/100:8.0f} 1/cm  rho1 {o['pops_tdh'][0]:.3f}/{o['pops_frozen'][0]:.3f}", flush=True)
