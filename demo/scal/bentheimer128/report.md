# scal_bentheimer128 -- SCAL pressure-ladder drainage + imbibition

- Geometry: `test/assets/bentheimer128.npy` + porous plate (2x2 holes, pitch 4, thick 3, theta 170 deg)
- Ladder: pc0=0.00561, dpc=0.0014, cap=0.02805; stall |dSw|<0.005 x2
- Drainage: 10 steps -> S_wi = 0.3053
- Imbibition: 15 steps -> S_or = 0.0021
- Max plate leak (red mass past plate, per block): 1.058e+03  (seal proof: ~0)
- Total scrubbed red (membrane production side): 4.611e+04
- Steps 1372000, 4.63 h this segment
- Curves: pc_steps.csv (the Pc-S_w loop), metrics.csv, ca_curve.svg; publication figs via publication_plot.py
