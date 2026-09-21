"""Measure the effective entry pressure of a narrow-pore porous plate (GPU, ~3 min).

Sharp-interface Young-Laplace predicts that a 2x2-voxel hole holds
~2*sigma*cos(theta_w)/1 ~ 0.1 LU, but the color-gradient interface is 4-5 cells
wide, so a 2-voxel hole cannot develop that barrier. The effective value is
therefore measured:

(a) Breach ramp, per hole size: open duct (no rock) so the red slug presses
    straight onto the plate -- the most severe case; ramp the inlet Pc in fine
    steps and record red mass past the plate; breach = first pc with > 1 voxel
    of red.
(b) Conductance: all-water domain, small dp across the plate -> finite, monotone
    water flux (the plate must not choke quasi-static drainage).

Writes output/holdoff.json; throat_ladder.py records the measured breach next to
the rock ladder for comparison with the ladder cap.
"""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import scal_demo

from bob import bc, color3d

OUT = Path(__file__).parent / "output"
BLOCK, N_BLOCKS = 500, 6
BREACH_MASS = 1.0  # one voxel's worth of red past the plate = breached


def build(all_blue, hole, pitch):
    duct = np.zeros((32, 32, 20), bool)  # open duct "rock"
    solid_np, reg = scal_demo.build_domain(duct, buf_in=6, buf_out=6, hole=hole, pitch=pitch)
    solid = jnp.asarray(solid_np)
    nw = color3d.wall_normals(solid)
    theta = jnp.asarray(scal_demo.theta_field_for(solid_np.shape, reg))
    omega = 1.0 / (3.0 * scal_demo.NU + 0.5)
    params = color3d.Params(
        omega=omega, omega2=omega, sigma=scal_demo.SIGMA, beta=scal_demo.BETA, theta=theta, mrt_chi=scal_demo.CHI
    )
    x_red = 0 if all_blue else reg["inbuf"][1]
    state = scal_demo.init_state_pressure(solid_np, x_red, 1.0)

    @jax.jit
    def run(state, rho_in, sa_red):
        def body(s, _):
            s = color3d.step(s, params, solid, nw=nw)
            s = bc.zou_he_inlet(s, solid=solid, rho_in=rho_in, sa_red=sa_red)
            s = bc.zou_he_outlet(s, solid=solid, rho_out=1.0, sa_red=0.0)
            return s, None

        return jax.lax.scan(body, state, None, length=BLOCK)[0]

    return state, run, solid_np, reg


def breach_ramp(hole, pitch):
    """First pc (0.010..0.060 in 0.0025 steps) at which red truly crosses the plate."""
    state, run, solid_np, reg = build(all_blue=False, hole=hole, pitch=pitch)
    pore = ~solid_np
    outbuf = pore.copy()
    outbuf[:, :, : reg["outbuf"][0]] = False
    trace = []
    for pc in np.arange(0.010, 0.0625, 0.0025):
        for _ in range(N_BLOCKS):
            state = run(state, jnp.asarray(1.0 + 3.0 * pc), jnp.asarray(1.0))
        rhoR, _ = color3d.densities(state)
        leak = float(np.asarray(rhoR)[outbuf].sum())
        trace.append({"pc": round(float(pc), 4), "leak": leak})
        print(f"  hole={hole} pc={pc:.4f}: red past plate = {leak:.3e}")
        if leak > BREACH_MASS:
            return trace[-1]["pc"], trace
    return None, trace


def conductance(hole, pitch):
    state, run, solid_np, reg = build(all_blue=True, hole=hole, pitch=pitch)
    holes = ~solid_np[:, :, reg["plate"][0]]
    cond = []
    for dp in (0.002, 0.004, 0.008):
        for _ in range(N_BLOCKS):
            state = run(state, jnp.asarray(1.0 + 3.0 * dp), jnp.asarray(0.0))
        u = np.asarray(color3d.velocity(state))
        flux = float(u[0][:, :, reg["plate"][0]][holes].sum())
        cond.append({"dp": dp, "flux": flux})
        print(f"  hole={hole} conductance dp={dp:.3f}: water flux = {flux:.4e}")
    return cond


def main():
    results = {}
    for hole in (2, 1):
        print(f"--- breach ramp: hole={hole} pitch=4 ---")
        breach, trace = breach_ramp(hole, 4)
        cond = conductance(hole, 4)
        fluxes = [c["flux"] for c in cond]
        results[str(hole)] = {
            "breach_pc": breach,
            "trace": trace,
            "conductance": cond,
            "cond_ok": bool(fluxes[0] > 0 and fluxes[0] < fluxes[1] < fluxes[2]),
        }
        print(f"  hole={hole}: breach at pc={breach}, conductance ok={results[str(hole)]['cond_ok']}")
    (OUT / "holdoff.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: {kk: v[kk] for kk in ("breach_pc", "cond_ok")} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
