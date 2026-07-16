/nobackup/nd276333/Workspace/Alternance/cdsaxs/src/roughness/
├── README.md
├── config.yaml
│
├── data/
│   ├── profiles/                  # generated w_n arrays, seed-named
│   ├── cdsaxs/
│   │   └── patterns/              # simulated diffraction patterns
│   └── cdsem/
│       ├── inputs/                # mesh .tri, silicon.mat, beam config (NEBULA inputs)
│       └── images/                # .tif images (provided, static) + metadata.json
│
├── src/
│   ├── profile/
│   │   └── generate.py            # PSD model (Eq.1) + generator (Eq.2) + self-consistency check
│   │
│   ├── cdsaxs/
│   │   ├── simulate.py            # form factor + stack model (Eq.3) -> diffraction pattern
│   │   └── extract.py             # extinction-point method + off-extinction method -> PSD
│   │
│   ├── cdsem/
│   │   ├── prepare.py             # mesh from w_n + silicon.mat + beam config -> data/cdsem/inputs/
│   │   ├── run_nebula.py          # calls installed NEBULA on data/cdsem/inputs/ -> data/cdsem/images/
│   │   └── extract.py             # image loading + in-house extractor + Verduin et al. extractor
│   │
│   └── shared/
│       ├── fitting.py             # single PSD-fit function, used by cdsaxs/extract.py and cdsem/extract.py
│       └── compare.py             # ground truth vs. recovered params, table/plot
│
├── scripts/
│   ├── run_all.py                 # --seed; default reviewer path, no NEBULA needed
│   └── run_nebula_pipeline.py     # --seed; optional, requires NEBULA installed
│
├── figures/
│
└── tests/
    ├── test_profile.py
    ├── test_cdsaxs.py
    └── test_cdsem.py