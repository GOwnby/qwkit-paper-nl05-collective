#!/bin/sh
# Regenerate every number and figure of the NL-05 manuscript (paper_nl05/). Total ~15 min on 2 cores.
set -e
cd "$(dirname "$0")/nl05"
python3 spectra.py         > spectra.log          # bare / Ando / local-field / TDH spectra vs density
python3 pathway.py         > pathway.log          # pathway factor F_exc: full TDH vs closed form
python3 survey.py          > survey.log           # design survey, kappa, efficiency merit
python3 kernel_geometry.py > kernel_geometry.log  # L_ij, c, kappa identity (needs survey.json)
python3 decompose.py       > decompose.log        # exact P2 decomposition; roles of the (12)x(23) kernel block
python3 double_count.py    > double_count.log     # measured energies used twice
python3 dynamic.py         > dynamic.log          # real-time TDH vs frozen-shift saturation (slowest step)
python3 dynamic2.py        > dynamic2.log         # saturation controls: quasi-static shift, bare real time, pumped probe
python3 masks.py           > masks.log            # kernel-block masks, rho2 -> 0 counterfactual, TW kappa
python3 zaluzny95.py       > zaluzny95.log        # benchmark against Zaluzny, PRB 51, 9757 (1995)
python3 krieguer24.py > krieguer24.log     # Krieguer-Todorov three-subband benchmark
python3 brackets.py       > brackets.log         # saturation: population brackets |A-B|/|A| and quasi-static energies (needs dynamic*.json)
python3 subband_mass.py   > subband_mass.log     # in-plane subband masses and E12 spread across the Fermi sea (Limitations)
python3 geometry_extra.py > geometry_extra.log   # harmonic L_1n/L_12, TW level positions (needs survey.json)
python3 kane_tdh.py       > kane_tdh.log         # two-band Kane dipoles and Hartree kernel in TDH; ground-state Kane check
python3 collect_numbers.py > numbers.log
python3 paper_figures.py
cd ../../paper_nl05 && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null && \
  pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
cp main.pdf ../NL05_paper.pdf
