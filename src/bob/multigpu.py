"""Multi-device domain decomposition via JAX sharding.

bob's solvers are pure ``jnp`` functions (roll-based streaming composed under
``jit``/``scan``), so multi-GPU parallelism needs NO solver changes: place the
arrays on a 1D device mesh sharded along z, call the SAME jitted ``step`` /
``drain``, and XLA's GSPMD partitioner compiles each ``jnp.roll`` across a
shard boundary into the one-plane halo exchange (CollectivePermute) an MPI LBM
code would write by hand. Numerics are unchanged — per-cell arithmetic is
local, so sharded results match single-device results to float64 roundoff
(pinned by ``test/test_multigpu.py``).

Decomposition axis: **z by default**, the leading spatial axis (third-from-last
of every 3D bob array), but ``shard(..., axis="y")`` / ``axis="x"`` slice the
other spatial axes for domains laid out differently. GSPMD is semantics-
preserving, so ANY axis gives bit-identical results (pinned per axis by
``test/test_multigpu.py``) — the choice is about performance and layout, not
correctness:

* walls on the sharded axis (e.g. the sealed z-walls of the wetting demos
  under z-sharding) are ordinary solid cells inside the first/last slab;
* plane BCs on the sharded axis (e.g. the porous x-inlet/outlet under
  x-sharding) still work — the boundary plane lives wholly on the first/last
  device and GSPMD keeps the update local — but BCs that reduce over that
  plane serialize on one device, so prefer an axis the BCs do NOT slice
  (for the porous x-flow pipeline: z or y).

Arrays with fewer than 3 dims (lattice constants, 2D fields) are replicated.

Usage — a drop-in around any existing 3D demo::

    m = multigpu.mesh(4)                       # first 4 GPUs (or CPU emulation)
    state, solid, nw = multigpu.shard((state, solid, nw), m)
    sharding = jax.tree_util.tree_map(lambda x: x.sharding, state)

    @partial(jax.jit, out_shardings=sharding, donate_argnums=0)
    def run_block(s):                          # Params holds strings -> close over it
        return porous3d.drain(s, params, solid, n_in, u_in, block, nw=nw)

    state = run_block(state)                   # loop; sharding propagates

In a dispatch loop like this, ``out_shardings`` + ``donate_argnums`` are not
optional niceties: they pin the output to the input's sharding AND layout, so
feeding the state back in is a jit cache hit. Without them XLA may hand back a
differently-laid-out output at large sizes (seen at 256^3 on 2 GPUs), each
call re-lowers, and jax 0.6.2 crashes in ``_resolve_in_layouts`` (``TypeError:
... Got UnspecifiedValue``). Donation also halves device memory. Benchmarks:
``demo/multi_gpu_scaling`` (256^3 drainage: 1.50x / 2.65x on 2 / 4 H100s,
bit-identical saturation).

**GSPMD slice-update pitfall (jax 0.6.2):** a slab ``.at[:, :, :, :n].set``
followed by another slice-update at the FAR end of the same sharded axis
compiles to a wrong program (no error is raised — the reservoir-inlet +
zero-gradient-outlet pair under ``axis="x"`` is off by 5.6e-2 at step 1, while
each BC alone is exact). Write per-plane/slab BCs as elementwise ``jnp.where``
on an index mask (+ ``jnp.roll`` for neighbor-plane copies) — see
``bc.inlet_reservoir`` / ``outlet_zero_gradient``; the parity cases
(``test/multigpu_parity.py``) pin every BC pattern on every axis.

Two shard_map halo backends beat plain GSPMD (``demo/multi_gpu_halo``, 256^3
drainage on 8xH100, fp32): ``halo_step`` — ONE packed deep exchange around the
unmodified ``color3d.step`` (1536 MLUPS, 63% efficiency) — and
``staged_halo_step`` — per-stage thin exchanges (phi at depth 4 + populations
at depth 1) so collision never runs on ghost planes (2169 MLUPS, 90%). Both
are z-decomposition only, bit-exact vs single-device (``test/test_multigpu.py``).

**``staged_halo_step`` is the DEFAULT multi-GPU backend**: every parallel run
script (scal, drainage, GDL, lenormand, trapping) builds
``step_fn = multigpu.staged_halo_step(mesh, params, step_solid, nw=nw)`` from
the UNSHARDED host statics BEFORE ``shard()`` and passes it to
``porous3d.drain(..., step_fn=...)`` or calls it in place of ``color3d.step``
in a custom body (BCs stay outside — x-plane ops on the GSPMD-sharded state).
A per-cell ``params.theta`` field (mixed wettability) is pre-padded alongside
solid/nw automatically. For ``inlet="piston"`` pass the SEALED mask
(``solid[:, :, 0] = True``) — the mask ``step`` sees inside ``drain``. The raw
step is bit-exact vs single-device (pinned incl. a theta field by
``test_multigpu.py``); over long fused BC+scan trajectories XLA's fusion can
seed 1-ULP floating-point reassociation differences that the contact-line
dynamics amplify chaotically, so long sharded trajectories are compared on
physical metrics, not bit identity. Plain GSPMD ``shard()`` with the unmodified
step remains the zero-wiring fallback and the any-axis (y/x) option.

The sharded extent must divide evenly by the device count (64 and 256 both do
for 1/2/4/8).
CPU emulation for tests/laptops: set ``XLA_FLAGS=--xla_force_host_platform_
device_count=N`` (before jax initializes) to split the host into N devices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from bob import color3d, hograd

AXIS = "z"  # mesh axis NAME (fixed); which ARRAY axis is sliced is shard()'s axis=
AXES = {"z": -3, "y": -2, "x": -1}  # spatial axis -> offset from the end of the shape


def devices(n=None, platform=None):
    """The first ``n`` JAX devices (all visible ones by default), optionally
    filtered by ``platform`` ("gpu"/"cpu"). ``CUDA_VISIBLE_DEVICES`` (GPU) and
    ``--xla_force_host_platform_device_count`` (CPU) control visibility."""
    devs = jax.devices(platform) if platform is not None else jax.devices()
    if n is not None:
        if n > len(devs):
            raise ValueError(f"asked for {n} devices but only {len(devs)} are visible: {devs}")
        devs = devs[:n]
    return devs


def mesh(n=None, platform=None):
    """1D device mesh over ``devices(n, platform)`` with axis name ``"z"``."""
    return Mesh(np.array(devices(n, platform)), axis_names=(AXIS,))


def _spec(ndim, axis="z"):
    """PartitionSpec slicing the requested spatial axis (z/y/x, counted from the
    END of the shape so leading population/vector dims never shift it); arrays
    below 3D are replicated."""
    if ndim < 3:
        return PartitionSpec()
    parts = [None] * ndim
    parts[ndim + AXES[axis]] = AXIS
    return PartitionSpec(*parts)


def shard(tree, m, axis="z"):
    """Shard every array leaf of ``tree`` along a spatial ``axis`` across mesh ``m``.

    ``tree`` is any pytree — a ``color3d.State``, a solid mask, ``nw`` wall
    normals, a force field, or a tuple of those. ``axis`` is ``"z"`` (default,
    third-from-last), ``"y"`` or ``"x"`` — see the module docstring for how to
    choose. Leaves with fewer than 3 dims are replicated. Raises when the
    sharded extent does not divide by the device count (GSPMD wants equal
    slabs; pad the domain or change the count)."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {sorted(AXES)}, got {axis!r}")
    ndev = len(m.devices.reshape(-1))

    def put(x):
        x = jnp.asarray(x)
        if x.ndim >= 3 and x.shape[AXES[axis]] % ndev != 0:
            raise ValueError(f"{axis} extent {x.shape[AXES[axis]]} of shape {x.shape} not divisible by {ndev} devices")
        return jax.device_put(x, NamedSharding(m, _spec(x.ndim, axis)))

    return jax.tree_util.tree_map(put, tree)


# ─── Halo-step path: packed ghost-plane exchange (shard_map) ───────────────────
#
# GSPMD emits one slice/concat + CollectivePermute per z-crossing ``jnp.roll`` —
# ~100+ small messages per default-stack drainage step (2x19 streaming rolls +
# the 4-stage CSF gather chain), which is the residual strong-scaling loss of
# ``demo/multi_gpu_scaling`` (XLA communication flags do not recover it). The halo
# path instead wraps the UNMODIFIED ``color3d.step`` in ``shard_map``: pad each
# device's z-slab with H ghost planes from its ring neighbors (2 ``lax.ppermute``
# per color per step, each one packed message carrying all H planes of all 19
# populations), run the step locally on the padded slab, discard the ghost
# planes. Ring wrap = global z-periodicity, so per interior cell the
# arithmetic graph is identical to the global program (parity pinned at 1e-12 by
# test/test_multigpu.py; ``required_halo`` is pinned MINIMAL by the halo_depth
# case). Cost: H redundant ghost planes recomputed per slab side — the
# message-count vs ghost-compute trade ``demo/multi_gpu_halo`` measures.


def required_halo(params):
    """Ghost-plane depth H so ``color3d.step`` on an H-padded slab reproduces the
    global step on the slab interior. Each neighbor-touching op invalidates one
    more ghost layer from the outside in; the deepest chain decides. With r the
    gradient stencil radius (``grad_order`` (2,4) -> 1, (2,6)/(2,8) -> 2):

    * CSF path (default): phi-extrapolate (1) + gradient (r) + normal-extrapolate
      (1) + curvature (r) + stream (1) = **3 + 2r** (default stack: 5).
      ``wall_grad="bulk"`` and ``halfway`` are shallower chains — no extra depth.
    * perturbation path (csf=False): fluid gradient (r) + stream (1) = r + 1;
      ``wall_grad="bulk"`` adds the gradient+neighbor-max magnitude chain (2).
    """
    r = hograd.radius(params.grad_order, 3)
    if params.csf:
        return 3 + 2 * r
    return (max(r, 2) if params.wall_grad == "bulk" else r) + 1


def _pad_static(x, ndev, halo):
    """Host-side pre-padding for per-geometry static fields (solid, nw): rebuild
    the global ``(..., nz, ny, nx)`` array as ``ndev`` concatenated padded slabs
    (local + 2*halo z-planes each, periodic wrap), so sharding the result along z
    hands every device its slab WITH ghost planes — zero per-step cost."""
    x = np.asarray(x)
    nz = x.shape[-3]
    local = nz // ndev
    idx = np.concatenate([np.arange(i * local - halo, (i + 1) * local + halo) for i in range(ndev)]) % nz
    return np.take(x, idx, axis=x.ndim - 3)


P4 = PartitionSpec(None, AXIS, None, None)  # (19|3, nz, ny, nx) — populations / vectors
P3 = PartitionSpec(AXIS, None, None)        # (nz, ny, nx) — masks / scalar fields


def _ring_pad(ndev, depth):
    """``pad(x)`` adding ``depth`` ghost z-planes per side from the ring neighbors
    (``lax.ppermute``); with one device the local wrap IS the global periodicity."""
    if ndev == 1:

        def _pad(x, d=depth):
            return jnp.concatenate([x[..., -d:, :, :], x, x[..., :d, :, :]], axis=-3)
    else:
        fwd = tuple((i, (i + 1) % ndev) for i in range(ndev))  # slab i -> i+1; wrap = z-periodicity
        bwd = tuple(((i + 1) % ndev, i) for i in range(ndev))

        def _pad(x, d=depth):
            lo = jax.lax.ppermute(x[..., -d:, :, :], AXIS, fwd)  # lower neighbor's top planes
            hi = jax.lax.ppermute(x[..., :d, :, :], AXIS, bwd)  # upper neighbor's bottom planes
            return jnp.concatenate([lo, x, hi], axis=-3)

    return _pad


def _pack_statics(m, params, solid, nw, ndev, depth):
    """Pre-pad + shard the per-geometry statics a halo backend closes over:
    the solid mask, optional wall normals, and — when ``params.theta`` is a
    per-cell (nz, ny, nx) field (the mixed-wettability demos) — the theta
    field itself, which stage 1 consumes at ghost depth alongside the padded
    arrays. Returns ``(statics, in_specs, params)`` with the array theta
    stripped out of ``params`` (re-injected per-slab inside the shard_map
    body; a scalar theta stays put — it broadcasts at any shape)."""
    statics = [jax.device_put(jnp.asarray(_pad_static(solid, ndev, depth)), NamedSharding(m, P3))]
    specs = [P3]
    if nw is not None:
        statics.append(jax.device_put(jnp.asarray(_pad_static(nw, ndev, depth)), NamedSharding(m, P4)))
        specs.append(P4)
    if params.theta is not None and np.ndim(params.theta) == 3:
        theta_p = _pad_static(np.asarray(params.theta), ndev, depth)
        statics.append(jax.device_put(jnp.asarray(theta_p), NamedSharding(m, P3)))
        specs.append(P3)
        params = params._replace(theta=None)  # only stage 1 reads theta; re-injected there
    return tuple(statics), tuple(specs), params


def halo_step(m, params, solid, nw=None, halo=None):
    """A packed-halo drop-in for ``color3d.step`` under z-sharding on mesh ``m``.

    Returns ``step_fn(state) -> state`` with ``params``/``solid``/``nw`` baked
    in — pass it to ``porous3d.drain(..., step_fn=...)`` (BCs stay outside: they
    are x-plane ops, communication-free under z-sharding). ``solid`` must be the
    mask ``step`` will see — for ``inlet="piston"`` pass the sealed mask
    (``solid.at[:, :, 0].set(True)``), matching what ``drain`` seals internally.
    ``halo`` overrides ``required_halo(params)`` (experiments only — parity needs
    the full depth). The caller shards the state as usual (``multigpu.shard``);
    statics are re-padded here, so pass the UNSHARDED host arrays."""
    ndev = int(m.devices.size)
    H = required_halo(params) if halo is None else int(halo)
    solid = np.asarray(solid)
    if solid.ndim != 3:
        raise ValueError(f"solid must be (nz, ny, nx), got shape {solid.shape}")
    nz = solid.shape[0]
    if nz % ndev:
        raise ValueError(f"z extent {nz} not divisible by {ndev} devices")
    local = nz // ndev
    if H > local:
        raise ValueError(
            f"halo depth {H} exceeds the {local}-plane slab of {ndev} devices "
            f"(the ring exchange only reaches adjacent devices) — fewer devices or a taller domain"
        )

    statics, specs, params = _pack_statics(m, params, solid, nw, ndev, H)
    has_nw, has_th = nw is not None, len(statics) == (3 if nw is not None else 2)
    _pad = _ring_pad(ndev, H)

    def _local(s, *stat):
        sol = stat[0]
        normals = stat[1] if has_nw else None
        p = params._replace(theta=stat[-1]) if has_th else params
        padded = color3d.State(_pad(s.fR), _pad(s.fB))
        out = color3d.step(padded, p, solid=sol, nw=normals)
        return color3d.State(out.fR[..., H:-H, :, :], out.fB[..., H:-H, :, :])

    inner = shard_map(_local, mesh=m, in_specs=(P4,) + specs, out_specs=P4, check_rep=False)
    return lambda state: inner(state, *statics)


# ─── Staged thin-halo path: per-stage exchanges instead of one deep one ─────────
#
# ``halo_step`` exchanges the full populations at depth ``required_halo`` (5) and
# recomputes the WHOLE step — MRT collision included — on every ghost plane; at 8
# devices that is 10 ghost planes on a 32-plane slab, a 31% compute tax
# (``demo/multi_gpu_halo``: 63% efficiency at 256^3 fp32). But the depth-5
# requirement is almost entirely the scalar phi chain (``color3d.surface_force``);
# collision is pointwise and streaming needs only depth 1. So exchange the
# 1-field rhoN at depth ``required_halo - 1`` + the 2x19-field post-collision
# populations at depth 1: ~4.5x less traffic, and ghost recompute shrinks to
# gradient arithmetic + one streaming plane. Same parity contract, pinned by the
# staged_* cases in test/test_multigpu.py.


def required_phi_halo(params):
    """Ghost depth of the rhoN exchange in ``staged_halo_step``: everything in
    ``color3d.surface_force`` — i.e. ``required_halo`` minus the single
    streaming layer that the depth-1 population exchange covers (default CSF
    stack: 4). Minimality is pinned by the ``staged_depth`` parity case."""
    return required_halo(params) - 1


def staged_halo_step(m, params, solid, nw=None, phi_halo=None):
    """Per-stage thin-halo drop-in for ``color3d.step`` under z-sharding — the
    low-ghost-compute alternative to ``halo_step`` (same contract: pass it to
    ``porous3d.drain(..., step_fn=...)``; BCs stay outside; for
    ``inlet="piston"`` pass the sealed mask; statics arrive UNSHARDED and are
    pre-padded here). ``phi_halo`` overrides ``required_phi_halo(params)``
    (experiments only — parity needs the full depth)."""
    ndev = int(m.devices.size)
    Hp = required_phi_halo(params) if phi_halo is None else int(phi_halo)
    solid = np.asarray(solid)
    if solid.ndim != 3:
        raise ValueError(f"solid must be (nz, ny, nx), got shape {solid.shape}")
    nz = solid.shape[0]
    if nz % ndev:
        raise ValueError(f"z extent {nz} not divisible by {ndev} devices")
    local = nz // ndev
    if Hp > local:
        raise ValueError(
            f"phi halo depth {Hp} exceeds the {local}-plane slab of {ndev} devices "
            f"(the ring exchange only reaches adjacent devices) — fewer devices or a taller domain"
        )

    statics, specs, params = _pack_statics(m, params, solid, nw, ndev, Hp)
    has_nw, has_th = nw is not None, len(statics) == (3 if nw is not None else 2)
    _pad = _ring_pad(ndev, Hp)

    # Design note: stage 3 deliberately uses the simple pad-then-compute form.
    # Splitting it into an interior stream (independent of the ppermutes) +
    # 3-plane edge strips, so XLA can overlap the depth-1 exchange with bulk
    # compute, is 15% SLOWER at 256^3 fp32 on 8 H100 (1851 vs 2169 MLUPS) — the
    # extra full-array concatenate + edge kernels cost more than the ~5-8%
    # communication they could hide.

    def _local(s, *stat):
        sol_p = stat[0]
        normals_p = stat[1] if has_nw else None
        p1 = params._replace(theta=stat[-1]) if has_th else params  # theta is stage-1-only
        sol = sol_p[Hp:-Hp]  # this device's local solid slab
        sol1 = sol_p if Hp == 1 else sol_p[Hp - 1 : -(Hp - 1)]  # local + 1 ghost plane each side
        rhoR, rhoB = s.fR.sum(0), s.fB.sum(0)
        rhoN = color3d.color_field(rhoR, rhoB)
        # stage 1 — deep exchange of the ONE scalar field the gradient chain needs
        F_p, G_p = color3d.surface_force(_pad(rhoN, Hp), p1, sol_p, normals_p)
        F_surf = None if F_p is None else F_p[:, Hp:-Hp]
        G = G_p[:, Hp:-Hp]
        # stage 2 — pointwise on local planes only (the expensive part: no ghosts)
        fR, fB = color3d.collide_recolor(s, rhoR, rhoB, params, F_surf, G)
        fR, fB = color3d.wall_restore(fR, fB, s, params, sol)
        # stage 3 — depth-1 exchange of the post-collision populations, stream, trim
        fR, fB = color3d.stream_bounce(_pad(fR, 1), _pad(fB, 1), params, sol1)
        return color3d.wall_seal(fR[:, 1:-1], fB[:, 1:-1], s, params, sol)

    inner = shard_map(_local, mesh=m, in_specs=(P4,) + specs, out_specs=P4, check_rep=False)
    return lambda state: inner(state, *statics)
