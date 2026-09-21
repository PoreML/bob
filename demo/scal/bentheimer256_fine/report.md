# scal_bent256_0014_fine -- SCAL pressure-ladder drainage + imbibition

- Geometry: `demo/scal/bentheimer256_fine/bent256_0014.npy` + porous plate (5x5 pores, pitch 8, thick 20, theta 150 deg)
- Ladder: pc0=0.00607, dpc=0.00038, cap=0.03037; stall |dSw|<0.005 x8
- Drainage: 35 steps -> S_wi = 0.3203
- Imbibition: 58 steps -> S_or = 0.2307
- Max plate leak (red mass past plate, per block): 2.538e-01  (seal proof: ~0)
- Total scrubbed red (membrane production side): 1.947e+02
- Steps 4676000, 39.52 h this segment
- Curves: pc_steps.csv (the Pc-S_w loop), metrics.csv, ca_curve.svg; publication figs via publication_plot.py
