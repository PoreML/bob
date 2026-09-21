"""Sharded-vs-single-device parity harness for ``bob.multigpu``.

NOT a pytest module (no ``test_`` prefix) — ``test_multigpu.py`` runs this in a
subprocess with ``XLA_FLAGS=--xla_force_host_platform_device_count=4`` and
``JAX_PLATFORMS=cpu``, because the device-count flag must be set before jax
initializes (impossible inside an already-running pytest process) and CPU
emulation makes the harness hardware-independent. Exit code 0 = parity holds on
2- and 4-device meshes along EVERY shard axis (z, y, x) for the requested
case; any assertion failure exits 1 with the mismatch printed.

Cases: ``single`` (single-phase D3Q19 forced duct), ``clamp`` (two-phase
drainage, velocity-reservoir inlet, wetting BC), ``piston`` (drainage with the
MF-LBM piston inlet + pinned-color outlet), ``velocity`` (rate-controlled
eqm velocity inlet + Zou-He pressure outlet — the production drainage /
scaling-benchmark driver), ``wetting`` (sessile droplet between sealed
z-walls — walls ON the sharded axis under axis="z"), ``zouhe`` (washburn
tube: per-color Zou-He pressure inlet AND outlet on x-planes — the BC axis
IS the sharded axis under axis="x"), ``pc`` (capillary-pressure ladder
pattern: fixed-pressure LAYER slabs at both x ends) — together they cover
streaming, MRT collision, CSF + akai wetting, and every roll/BC pattern the
demos use, with the sliced axis crossing walls, inlets, and periodic faces.

``clamp`` under axis="x" is the regression pin for the jax 0.6.2 GSPMD
slice-update bug: a slab ``.at[:, :, :, :n].set`` followed by a plane
``.at[:, :, :, -1].set`` on the SAME sharded axis compiles to a wrong program
(5.6e-2 at step 1); ``inlet_reservoir`` / ``outlet_zero_gradient`` (and the
capillary_pressure layer BCs) are therefore written as jnp.where/jnp.roll.
"""

import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from bob import bc, color3d, lbm3d, multigpu, porous3d

TOL = 1e-12  # float64 roundoff; sharding must not change per-cell arithmetic
AXES = ("z", "y", "x")


def _duct_case():
    """Small drainage duct: (16, 16, 32), red slug at the inlet, wetting walls."""
    solid = porous3d.duct(16, 16, 32)
    state = porous3d.init_drainage(16, 16, 32, n_red=8)
    params = color3d.Params(omega=1.0, sigma=0.05, beta=0.7, theta=120.0)
    nw = color3d.wall_normals(solid)
    return state, params, solid, nw


def run_single(m, axis):
    """Single-phase forced duct flow: collide -> stream -> bounce-back, 40 steps."""
    solid = porous3d.duct(16, 16, 32)
    f = lbm3d.equilibrium(jnp.ones((16, 16, 32)), jnp.zeros((3, 16, 16, 32)))
    force = jnp.array([1e-5, 0.0, 0.0])

    @jax.jit
    def run(f):
        def body(f, _):
            f = lbm3d.collide(f, omega=1.0, force=force)
            f = lbm3d.stream(f)
            return lbm3d.bounce_back(f, solid), None

        return jax.lax.scan(body, f, None, length=40)[0]

    ref = np.asarray(run(f))
    (f_s,) = multigpu.shard((f,), m, axis=axis)
    got = np.asarray(run(f_s))
    return np.abs(got - ref).max()


def _state_err(got, ref):
    return max(np.abs(np.asarray(got.fR) - np.asarray(ref.fR)).max(), np.abs(np.asarray(got.fB) - np.asarray(ref.fB)).max())


def _run_drain(m, axis, inlet, outlet_sa_red):
    state, params, solid, nw = _duct_case()

    @jax.jit
    def run(s, sol, normals):  # params/inlet closed over: Params holds strings (demo convention)
        return porous3d.drain(s, params, sol, 4, 0.01, 30, nw=normals, inlet=inlet, outlet_sa_red=outlet_sa_red)

    ref = run(state, solid, nw)
    state_s, solid_s, nw_s = multigpu.shard((state, solid, nw), m, axis=axis)
    got = run(state_s, solid_s, nw_s)
    return _state_err(got, ref)


def run_wetting(m, axis):
    """Sessile red half-ball between sealed z-walls (wetting pattern): walls on
    the z faces, x/y periodic — under axis='z' the walls sit on the sharded axis."""
    nz, ny, nx = 16, 24, 24
    solid = np.zeros((nz, ny, nx), bool)
    solid[0] = solid[-1] = True
    nw = color3d.wall_normals(solid)
    params = color3d.Params(omega=1.0, sigma=0.02, beta=0.95, theta=60.0)
    zz, yy, xx = jnp.meshgrid(jnp.arange(nz), jnp.arange(ny), jnp.arange(nx), indexing="ij")
    red = ((xx - (nx - 1) / 2.0) ** 2 + (yy - (ny - 1) / 2.0) ** 2 + (zz - 1.0) ** 2 <= 5.0**2) & (zz >= 1)
    from bob import d3q19

    W = d3q19.W[:, None, None, None]
    state = color3d.State(W * jnp.where(red, 1.0, 0.0)[None], W * jnp.where(red, 0.0, 1.0)[None])
    solid_j = jnp.asarray(solid)

    @jax.jit
    def run(s, sol, normals):
        return jax.lax.scan(lambda a, _: (color3d.step(a, params, solid=sol, nw=normals), None), s, None, length=30)[0]

    ref = run(state, solid_j, nw)
    state_s, solid_s, nw_s = multigpu.shard((state, solid_j, nw), m, axis=axis)
    got = run(state_s, solid_s, nw_s)
    return _state_err(got, ref)


def run_zouhe(m, axis):
    """Washburn tube (washburn pattern): cylinder along x, frictionless reservoir
    chambers, per-color Zou-He PRESSURE inlet and outlet on the x faces — under
    axis='x' both BC planes sit on the sharded axis."""
    nz, ny, nx, R, res = 8, 8, 32, 3, 4
    yc, zc = (ny - 1) / 2.0, (nz - 1) / 2.0
    zz2, yy2 = np.indices((nz, ny))
    tube = ((yy2 - yc) ** 2 + (zz2 - zc) ** 2) >= R**2
    solid = np.zeros((nz, ny, nx), bool)
    solid[:, :, res : nx - res] = tube[:, :, None]
    nw = color3d.wall_normals(solid)
    params = color3d.Params(omega=1.0, sigma=1.0 / 45, beta=0.95, theta=30.0, omega2=1.0 / 0.6)
    state = porous3d.init_drainage(nz, ny, nx, n_red=res)
    solid_j = jnp.asarray(solid)

    @jax.jit
    def run(s, sol, normals):
        def body(a, _):
            a = color3d.step(a, params, solid=sol, nw=normals)
            a = bc.zou_he_inlet(a, solid=sol, rho_in=1.0, sa_red=1.0)
            a = bc.zou_he_outlet(a, solid=sol, rho_out=1.0)
            return a, None

        return jax.lax.scan(body, s, None, length=30)[0]

    ref = run(state, solid_j, nw)
    state_s, solid_s, nw_s = multigpu.shard((state, solid_j, nw), m, axis=axis)
    got = run(state_s, solid_s, nw_s)
    return _state_err(got, ref)


def run_pc(m, axis):
    """Capillary-pressure ladder pattern (demo/capillary_pressure): fixed-pressure
    pure-red layers at x<2 and pure-blue layers at x>=nx-2, both slab updates on
    the x axis — the second opposite-end slab of the GSPMD pitfall."""
    from bob import d3q19

    state, params, solid, nw = _duct_case()
    W = d3q19.W[:, None, None, None]
    nx = state.fR.shape[3]
    xidx = jnp.arange(nx)
    solid_j = jnp.asarray(solid)

    @jax.jit
    def run(s, sol, normals):
        def body(a, _):
            a = color3d.step(a, params, solid=sol, nw=normals)
            a = color3d.State(jnp.where(xidx < 2, W * 1.0, a.fR), jnp.where(xidx < 2, 0.0, a.fB))
            a = color3d.State(jnp.where(xidx >= nx - 2, 0.0, a.fR), jnp.where(xidx >= nx - 2, W * 0.98, a.fB))
            return a, None

        return jax.lax.scan(body, s, None, length=30)[0]

    ref = run(state, solid_j, nw)
    state_s, solid_s, nw_s = multigpu.shard((state, solid_j, nw), m, axis=axis)
    got = run(state_s, solid_s, nw_s)
    return _state_err(got, ref)


def _tall_duct_case():
    """Taller-z drainage duct (40, 16, 24): the halo-step cases pad each local
    slab with ``required_halo`` ghost planes (5 for the default CSF stack), and a
    ring exchange only reaches adjacent devices, so nz/ndev must be >= 5 on the
    4-device mesh."""
    solid = porous3d.duct(40, 16, 24)
    state = porous3d.init_drainage(40, 16, 24, n_red=6)
    params = color3d.Params(omega=1.0, sigma=0.05, beta=0.7, theta=120.0)
    nw = color3d.wall_normals(solid)
    return state, params, solid, nw


def _run_drain_halo(m, inlet, outlet_sa_red, halo=None):
    """Halo-step drain (packed shard_map exchange) vs single-device drain."""
    state, params, solid, nw = _tall_duct_case()

    @jax.jit
    def ref_run(s, sol, normals):
        return porous3d.drain(s, params, sol, 4, 0.01, 30, nw=normals, inlet=inlet, outlet_sa_red=outlet_sa_red)

    ref = ref_run(state, solid, nw)
    # halo_step must see the mask step() sees — drain seals x=0 itself for the piston
    step_solid = jnp.asarray(solid).at[:, :, 0].set(True) if inlet == "piston" else solid
    step_fn = multigpu.halo_step(m, params, step_solid, nw=nw, halo=halo)
    state_s, solid_s, nw_s = multigpu.shard((state, solid, nw), m)

    @jax.jit
    def halo_run(s, sol, normals):
        return porous3d.drain(
            s, params, sol, 4, 0.01, 30, nw=normals, inlet=inlet, outlet_sa_red=outlet_sa_red, step_fn=step_fn
        )

    got = halo_run(state_s, solid_s, nw_s)
    return _state_err(got, ref)


def _run_halo_depth_pin(m):
    """Pin that required_halo is exact for the default CSF stack: depth 5 must
    hold parity (checked by the halo cases) and depth 4 must BREAK it — i.e. the
    ghost-plane analysis is minimal, not merely sufficient."""
    _, params, _, _ = _tall_duct_case()
    need = multigpu.required_halo(params)
    assert need == 5, f"default CSF stack should need 5 ghost planes, required_halo says {need}"
    err = _run_drain_halo(m, "velocity", 0.0, halo=need - 1)
    assert err >= TOL, f"halo={need - 1} unexpectedly held parity ({err:.3e}) — required_halo overestimates"
    return 0.0


def _run_drain_staged(m, inlet, outlet_sa_red, phi_halo=None):
    """Staged thin-halo drain (phi at depth required_phi_halo + populations at
    depth 1) vs single-device drain."""
    state, params, solid, nw = _tall_duct_case()

    @jax.jit
    def ref_run(s, sol, normals):
        return porous3d.drain(s, params, sol, 4, 0.01, 30, nw=normals, inlet=inlet, outlet_sa_red=outlet_sa_red)

    ref = ref_run(state, solid, nw)
    step_solid = jnp.asarray(solid).at[:, :, 0].set(True) if inlet == "piston" else solid
    step_fn = multigpu.staged_halo_step(m, params, step_solid, nw=nw, phi_halo=phi_halo)
    state_s, solid_s, nw_s = multigpu.shard((state, solid, nw), m)

    @jax.jit
    def staged_run(s, sol, normals):
        return porous3d.drain(
            s, params, sol, 4, 0.01, 30, nw=normals, inlet=inlet, outlet_sa_red=outlet_sa_red, step_fn=step_fn
        )

    got = staged_run(state_s, solid_s, nw_s)
    return _state_err(got, ref)


def _run_drain_staged_theta(m):
    """Per-cell theta FIELD under the halo backends (the scal/gdl/drainage
    mixed-wettability pattern). The field varies along x (rock vs plate zones)
    AND along the sharded z axis, so slab boundaries cut straight through it —
    correctness requires the theta ghost padding to match solid/nw exactly.

    The pin is the RAW STEP: one application of each backend must equal the
    single-device ``color3d.step`` to bit exactness (observed 0.0). A full
    30-step ``drain`` trajectory is additionally bounded at 1e-8, NOT 1e-12:
    XLA fuses the step with the scan/BC context and may reassociate one ULP
    (a bare 30-step scan already differs by 4e-16 from the composed program,
    field or scalar theta alike), and the mixed-wettability interface
    dynamics amplify that seed — a chaotic last-bit divergence at the contact
    line, so the trajectory is judged on bounds, not bit identity."""
    state, params, solid, nw = _tall_duct_case()
    nz, _ny, nx = solid.shape
    theta = np.full(solid.shape, 120.0)
    theta[:, :, nx // 2 :] = 150.0   # mixed wettability along the flow axis
    theta[: nz // 2] += 5.0          # and across the sharded axis
    params = params._replace(theta=jnp.asarray(theta))

    ref1 = color3d.step(state, params, solid=solid, nw=nw)
    state_s, solid_s, nw_s = multigpu.shard((state, solid, nw), m)
    worst = 0.0
    for backend in (multigpu.staged_halo_step, multigpu.halo_step):
        step_fn = backend(m, params, solid, nw=nw)
        worst = max(worst, _state_err(step_fn(state_s), ref1))  # bit-exact contract

    @jax.jit
    def ref_run(s, sol, normals):
        return porous3d.drain(s, params, sol, 4, 0.01, 30, nw=normals, inlet="velocity", outlet_sa_red=0.0)

    step_fn = multigpu.staged_halo_step(m, params, solid, nw=nw)

    @jax.jit
    def staged_run(s, sol, normals):
        return porous3d.drain(
            s, params, sol, 4, 0.01, 30, nw=normals, inlet="velocity", outlet_sa_red=0.0, step_fn=step_fn
        )

    drift = _state_err(staged_run(state_s, solid_s, nw_s), ref_run(state, solid, nw))
    assert drift < 1e-8, f"theta-field drain trajectory drifted {drift:.3e} (ULP-amplification bound 1e-8)"
    return worst


def _run_staged_depth_pin(m):
    """Pin that required_phi_halo is exact: depth 4 holds parity (staged cases),
    depth 3 must BREAK it — the phi-chain analysis is minimal, not merely
    sufficient."""
    _, params, _, _ = _tall_duct_case()
    need = multigpu.required_phi_halo(params)
    assert need == 4, f"default CSF stack should need 4 phi ghost planes, required_phi_halo says {need}"
    err = _run_drain_staged(m, "velocity", 0.0, phi_halo=need - 1)
    assert err >= TOL, f"phi_halo={need - 1} unexpectedly held parity ({err:.3e}) — required_phi_halo overestimates"
    return 0.0


CASES = {
    "single": run_single,
    "clamp": lambda m, axis: _run_drain(m, axis, "clamp", None),
    "piston": lambda m, axis: _run_drain(m, axis, "piston", 0.0),
    "velocity": lambda m, axis: _run_drain(m, axis, "velocity", 0.0),
    "wetting": run_wetting,
    "zouhe": run_zouhe,
    "pc": run_pc,
    # halo-step cases (z-decomposition only — axis loop reduced to "z" in main)
    "halo_clamp": lambda m, axis: _run_drain_halo(m, "clamp", None),
    "halo_piston": lambda m, axis: _run_drain_halo(m, "piston", 0.0),
    "halo_velocity": lambda m, axis: _run_drain_halo(m, "velocity", 0.0),
    "halo_depth": lambda m, axis: _run_halo_depth_pin(m),
    "staged_clamp": lambda m, axis: _run_drain_staged(m, "clamp", None),
    "staged_piston": lambda m, axis: _run_drain_staged(m, "piston", 0.0),
    "staged_velocity": lambda m, axis: _run_drain_staged(m, "velocity", 0.0),
    "staged_theta": lambda m, axis: _run_drain_staged_theta(m),
    "staged_depth": lambda m, axis: _run_staged_depth_pin(m),
}


def main():
    case = sys.argv[1]
    ndev = len(jax.devices())
    assert ndev >= 4, f"expected >= 4 emulated devices, got {ndev} (XLA_FLAGS not applied?)"
    axes = ("z",) if case.startswith(("halo", "staged")) else AXES  # halo/staged cases are z-decomposition only
    for axis in axes:
        for n in (2, 4):
            err = CASES[case](multigpu.mesh(n), axis)
            print(f"{case}: {n} devices, axis={axis}, max |sharded - single| = {err:.3e}")
            assert err < TOL, f"{case} diverged on {n} devices along {axis}: {err:.3e} >= {TOL}"
    # sanity: over-asking must fail loudly, indivisible extents and bad axes rejected
    try:
        multigpu.mesh(ndev + 1)
    except ValueError:
        pass
    else:
        raise AssertionError("mesh(too_many) did not raise")
    for shape, axis in (((3, 15, 8, 8), "z"), ((3, 8, 8, 15), "x")):
        try:
            multigpu.shard(jnp.zeros(shape), multigpu.mesh(2), axis=axis)
        except ValueError:
            pass
        else:
            raise AssertionError(f"indivisible {axis} extent did not raise")
    try:
        multigpu.shard(jnp.zeros((8, 8, 8)), multigpu.mesh(2), axis="w")
    except ValueError:
        pass
    else:
        raise AssertionError("bad axis name did not raise")
    print(f"{case}: OK")


if __name__ == "__main__":
    main()
