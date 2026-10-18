"""Collect the numbers quoted in the NL-05 manuscript into numbers.json (from the study JSON outputs)."""
import os, json, numpy as np
here = os.path.dirname(os.path.abspath(__file__))
L = lambda f: json.load(open(os.path.join(here, f + ".json")))
SP, PW, DY, SV, KG, DC, DE, MK, D2 = (L(f) for f in ["spectra", "pathway", "dynamic", "survey", "kernel_geometry", "double_count", "decompose", "masks", "dynamic2"])
out_zal = L("zaluzny95")
out = {"spectra": [], "pathway": [], "dynamic": [], "survey": SV, "geometry": KG, "double_count": DC, "decompose": DE, "masks": MK,
       "dynamic_controls": {k: v for k, v in D2.items() if k != "probe"},
       "probe_peaks_meV": {ns: [(r["I_MWcm2"], r["peak_meV"]) for r in d["runs"]] for ns, d in D2["probe"].items()}}
for o in SP:
    pk = lambda k: float(np.abs(np.array(o[k]) @ [1, 1j]).max())
    hw = np.array(o["hw"])
    out["spectra"].append(dict(name=o["name"], ns=o["ns"], E12_meV=1e3 * (o["E"][o["i2"]] - o["E"][0]),
                               shift_meV=1e3 * (o["Et"][o["i2"]][0] - (o["E"][o["i2"]] - o["E"][0])), Leff_A=o["Leff_A"],
                               chi2_nm={k: pk(k) / 1e3 for k in ["bare", "ando", "tdh", "lf"]},
                               tdh_over_ando=pk("tdh") / pk("ando"), lf_over_ando=pk("lf") / pk("ando"),
                               a1_peak_cm={k: max(o["a1_" + k]) / 100 for k in ["bare", "ando", "tdh"]},
                               a1_tdh_peak_meV=1e3 * float(hw[np.argmax(o["a1_tdh"])])))
for o in PW:
    out["pathway"].append(dict(ns=o["ns"], kappa_exc=o["kappa_exc"], D12_meV=o["D12_meV"],
                               max_dev=[max(abs(r["Fexc"][i] - r["Fexc_rwa"][i]) for r in o["rows"]) for i in (0, 1)]))
for ns in sorted({r["ns"] for r in DY}):
    rs = sorted([r for r in DY if r["ns"] == ns], key=lambda r: r["I_MWcm2"])
    out["dynamic"].append(dict(ns=ns, I=[r["I_MWcm2"] for r in rs],
                               ratio=[abs(complex(*r["chi2_tdh"])) / abs(complex(*r["chi2_frozen"])) for r in rs],
                               chi2_tdh_nm=[abs(complex(*r["chi2_tdh"])) / 1e3 for r in rs],
                               chi2_frozen_nm=[abs(complex(*r["chi2_frozen"])) / 1e3 for r in rs],
                               alpha_ratio=[r["alpha_tdh"] / r["alpha_frozen"] for r in rs], pops=[r["pops_tdh"] for r in rs]))
out["zaluzny95"] = {k: {kk: vv for kk, vv in v.items() if kk != "peak_vs_V"} for k, v in out_zal["res"].items()}
out["zaluzny95_V110"] = out_zal["V110"]
# ---- audit additions: derived numbers quoted in the text --------------------------------------------------------
BR, SM, GX, KT = (L(f) for f in ["brackets", "subband_mass", "geometry_extra", "kane_tdh"])
out["brackets"] = BR; out["subband_mass"] = SM; out["geometry_extra"] = GX; out["kane_tdh"] = KT
dag = lambda o: o["ns"] == 2e12 and o["name"] in ("Vurgaftman SI 26/19.5/57", "ADQW 50/15/30")   # box-dependent rows
Mr = lambda o, k: o["res"][k]["M"] / o["res"]["bare"]["M"]
sv = [o for o in SV if "skipped" not in o]
A = {}
# Sec. 7 / Discussion: merit of energy substitution and LF relative to TDH (deviation X/TDH - 1, the paper's convention)
A["merit_ando_dev_le1e12_max"] = max(abs(Mr(o, "ando") / Mr(o, "tdh") - 1) for o in sv if o["ns"] <= 1e12)
A["merit_ando_dev_range_excl_ADQW2e12"] = [f(abs(Mr(o, "ando") / Mr(o, "tdh") - 1) for o in sv if not dag(o) and not (o["ns"] == 2e12 and o["name"].startswith("N1 ADQW"))) for f in (min, max)]
A["merit_tdh_over_ando_max"] = max(Mr(o, "tdh") / Mr(o, "ando") for o in sv if not dag(o))
A["merit_tdh_over_lf_max"] = max(Mr(o, "tdh") / Mr(o, "lf") for o in sv if not dag(o))
A["merit_lf_dev_excl_TW2e12_WG2e12_max"] = max(abs(Mr(o, "lf") / Mr(o, "tdh") - 1) for o in sv if not dag(o) and not (o["ns"] == 2e12 and (o["name"].startswith("N1 triple") or "WG" in o["name"])))
A["merit_tdh_range"] = [f(Mr(o, "tdh") for o in sv if not dag(o)) for f in (min, max)]
# Discussion: pump absorption of energy substitution vs TDH up to 1e12 (peak for ADQW/TW from spectra; at the |chi2| maximum for all designs)
A["a1_peak_ando_dev_le1e12_ADQW_TW_max"] = max(s["a1_peak_cm"]["ando"] / s["a1_peak_cm"]["tdh"] - 1 for s in out["spectra"] if s["ns"] <= 1e12)
A["a1_oppoint_ando_dev_le1e12_max"] = max(abs(o["res"]["ando"]["a1"] / o["res"]["tdh"]["a1"] - 1) for o in sv if o["ns"] <= 1e12)
A["chi2_tdh_over_ando_le1e12_range"] = [f([s["tdh_over_ando"] for s in out["spectra"] if s["ns"] <= 1e12] + [o["res"]["tdh"]["chi2_nm"] / o["res"]["ando"]["chi2_nm"] for o in sv if o["ns"] <= 1e12]) for f in (min, max)]
# Sec. 4: closed-form F_exc vs full TDH, max |deviation| of real and imaginary parts
A["Fexc_dev_le1e12"] = max(max(p["max_dev"]) for p in out["pathway"] if p["ns"] <= 1e12)
A["Fexc_dev_2e12"] = max(next(p for p in out["pathway"] if p["ns"] == 2e12)["max_dev"])
# Sec. 8: saturation at 5e11 and 2e12
dy = {d["ns"]: d for d in out["dynamic"]}
qs = lambda ns: [abs(complex(*q["chi2"])) for q in D2["quasi_static"][str(ns)]]
bc = lambda ns: [abs(complex(*o["chi2_rt"])) / abs(complex(*o["chi2_rate"])) for o in D2["bare"] if o["ns"] == ns]
fz = lambda ns: [x * 1e3 for x in dy[ns]["chi2_frozen_nm"]]
I = dy[5e11]["I"]; hi = [i for i, x in enumerate(I) if x >= 3]; lo = [i for i, x in enumerate(I) if x <= 3]
A["sat5e11_tdh_frozen_dev_le3"] = max(abs(dy[5e11]["ratio"][i] - 1) for i in lo)
A["sat5e11_tdh_frozen_dev_all"] = max(abs(r - 1) for r in dy[5e11]["ratio"])
A["sat5e11_alpha_dev_all"] = max(abs(r - 1) for r in dy[5e11]["alpha_ratio"])
prod = [qs(5e11)[i] / fz(5e11)[i] * bc(5e11)[i] for i in hi]
A["sat5e11_qs_over_frozen_3to30"] = [qs(5e11)[i] / fz(5e11)[i] for i in hi]
A["sat5e11_bare_control_3to30"] = [bc(5e11)[i] for i in hi]
A["sat5e11_product_3to30"] = prod
A["sat5e11_tdh_over_frozen_3to30"] = [dy[5e11]["ratio"][i] for i in hi]
A["sat5e11_product_vs_tdh_dev_max"] = max(abs(p / dy[5e11]["ratio"][i] - 1) for p, i in zip(prod, hi))
A["sat2e12_qs_over_tdh_le3"] = [qs(2e12)[i] / (dy[2e12]["chi2_tdh_nm"][i] * 1e3) for i in lo]
A["sat2e12_qs_over_tdh_10_30"] = [qs(2e12)[i] / (dy[2e12]["chi2_tdh_nm"][i] * 1e3) for i in hi[1:]]
A["sat2e12_qs_over_frozen_3to30"] = [qs(2e12)[i] / fz(2e12)[i] for i in hi]
rows = BR["2000000000000.0"]["rows"]
A["sat2e12_brackets_3to30"] = {k: [o[k] for o in rows if o["I_MWcm2"] >= 3] for k in ("frozen_AmB_over_A", "qs_AmB_over_A", "bracket_ratio", "E12_meV", "E23_meV", "Et12_meV", "Et13_meV")}
A["sat2e12_equilibrium"] = BR["2000000000000.0"]["equilibrium"]
# Sec. 6 / 7: double counting
for d in DC:
    sp_ = next(s for s in out["spectra"] if s["ns"] == d["ns"] and (s["name"] == "ADQW") == d["name"].startswith("N1 ADQW"))
    A[f"double_count_{sp_['name']}_{d['ns']:.0e}"] = dict(vs_lf=d["chi2_dc"][0] / 1e3 / sp_["chi2_nm"]["lf"] - 1, vs_tdh=d["chi2_dc"][0] / d["chi2_tdh"][0] - 1,
                                                       line_shift_meV=1e3 * (d["lin_peak_dc"] - d["lin_peak_tdh"]))
# Benchmarks
A["zaluzny_full_over_selfcross"] = out_zal["res"]["full"]["max"] / out_zal["res"]["self+cross"]["max"]
# Limitations / geometry
A["harmonic_L_ratio_range"] = [f(h["ratio"] for h in GX["harmonic"]) for f in (min, max)]
A["TW_Et23_1e12_2e12_meV"] = [t["Et23_meV"] for t in GX["tw"] if t["ns"] in (1e12, 2e12)]
A["TW_level4_above_downhill_2e12_meV"] = next(t["E4_above_downhill_meV"] for t in GX["tw"] if t["ns"] == 2e12)
A["TW_E5_minus_E2_4e12_meV"] = next(t["E5_minus_E2_meV"] for t in GX["tw"] if t["ns"] == 4e12)
A["mass_m2_over_m1_minus1_2e12"] = next(s["dm21_rel"] for s in SM if s["ns"] == 2e12)
A["E12_spread_fermi_sea_2e12_meV"] = next(s["spread_E12_meV"] for s in SM if s["ns"] == 2e12)
# Kane: ground-state check (Sec. 2) and Kane kernel in TDH
A["kane_ground_chi2_change"] = [f(g["chi2_ratio"] - 1 for g in KT["ground"]) for f in (min, max)]
A["kane_two_band_overlap_max"] = max([g["two_band_overlap"] for g in KT["ground"]] + [o["two_band_overlap"] for o in KT["tdh"]])
kk = KT["tdh"]
A["kane_kernel_pump_shift_change"] = [f(o["kane"]["D_pump_meV"] / o["baseline"]["D_pump_meV"] - 1 for o in kk) for f in (min, max)]
A["kane_kernel_Et_pump_change_meV"] = [f(o["kane"]["Et_pump_meV"] - o["baseline"]["Et_pump_meV"] for o in kk) for f in (min, max)]
A["kane_kernel_Et_harm_change_meV"] = [f(o["kane"]["Et_harm_meV"] - o["baseline"]["Et_harm_meV"] for o in kk) for f in (min, max)]
A["kane_kernel_tdh_chi2_change_fixed_Z"] = [f(o["kane"]["tdh_chi2_nm"] / o["kane_Z"]["tdh_chi2_nm"] - 1 for o in kk) for f in (min, max)]
A["kane_full_tdh_chi2_change"] = [f(o["kane"]["tdh_chi2_nm"] / o["baseline"]["tdh_chi2_nm"] - 1 for o in kk) for f in (min, max)]
A["kane_tdh_over_ando_change"] = [f(o["kane"]["tdh_over_ando"] - o["baseline"]["tdh_over_ando"] for o in kk) for f in (min, max)]
A["kane_abs_peak_shift_meV"] = max(max(abs(a - b) for a, b in zip(o["kane"]["abs_peaks_meV"], o["baseline"]["abs_peaks_meV"])) for o in kk)
out["audit"] = A
json.dump(out, open(os.path.join(here, "numbers.json"), "w"), indent=1)
for s in out["spectra"]:
    print(f"{s['name']:4s} {s['ns']:.0e} E12={s['E12_meV']:.1f} shift={s['shift_meV']:.1f} chi2 " + "/".join(f"{v:.0f}" for v in s["chi2_nm"].values())
          + f" TDH/Ando={s['tdh_over_ando']:.2f} LF/Ando={s['lf_over_ando']:.2f} a1 " + "/".join(f"{v:.0f}" for v in s["a1_peak_cm"].values()) + f" @{s['a1_tdh_peak_meV']:.1f}")
for d in out["dynamic"]:
    print(f"dynamic {d['ns']:.0e}: ratio " + " ".join(f"{x:.2f}" for x in d["ratio"]) + " | alpha ratio " + " ".join(f"{x:.2f}" for x in d["alpha_ratio"]))
rnd = lambda v: float(f'{v:.4g}') if isinstance(v, float) else ([rnd(x) for x in v] if isinstance(v, list) else ({k: rnd(x) for k, x in v.items()} if isinstance(v, dict) else v))
for k, v in out["audit"].items():
    print(f"audit {k}: {rnd(v)}")
