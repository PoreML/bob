# scal_bent128_fine -- SCAL pressure-ladder drainage + imbibition

- Geometry: `test/assets/bentheimer128.npy` + porous plate (2x2 holes, pitch 4, thick 3, theta 170 deg)
- Ladder: pc0=0.00561, dpc=0.00035, cap=0.02805; stall |dSw|<0.005 x8
- Drainage: 23 steps -> S_wi = 0.3436
- Imbibition: 36 steps -> S_or = 0.0044
- Max plate leak (red mass past plate, per block): 3.908e+02  (seal proof: ~0)
- Total scrubbed red (membrane production side): 1.434e+05
- Steps 3850000, 7.63 h this segment
- Curves: pc_steps.csv (the Pc-S_w loop), metrics.csv, ca_curve.svg; publication figs via publication_plot.py
