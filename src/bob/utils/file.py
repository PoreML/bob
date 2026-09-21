"""Selective, time-series field I/O: HDF5 store + XDMF sidecar.

Distinct from the resume checkpoint (``bob.utils.stats``, which dumps raw
populations): this writes *derived* physical fields (``phi``, velocity,
pressure, density) for a whole simulation into one compact ``run.h5`` that
reads back partially into numpy and opens in ParaView via ``run.xdmf``.

- ``FieldWriter``  : append per-step groups into one file; rebuild the XDMF
  temporal collection after each append (watchable mid-run).
- ``FieldReader``  : load only the requested field / step / slice.
- ``FIELDS``       : name -> deriver registry over a ``color3d.State``.

Float fields default to float32; the rock mask is gzip'd uint8 (ParaView-
readable). The ``u`` vector field is stored ``(nz,ny,nx,3)`` (XDMF order).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from bob import color3d
from bob.color3d import color_field
from bob.d3q19 import CS2


class _Ctx:
    """Memoizes the shared subexpressions (densities, velocity) across a write."""

    def __init__(self, state):
        self.state = state
        self._d = None
        self._u = None

    @property
    def densities(self):
        if self._d is None:
            self._d = color3d.densities(self.state)
        return self._d

    @property
    def velocity(self):
        if self._u is None:
            self._u = color3d.velocity(self.state)
        return self._u


FIELDS = {
    "phi": lambda c: color_field(*c.densities),
    "rho": lambda c: c.densities[0] + c.densities[1],
    "rhoR": lambda c: c.densities[0],
    "rhoB": lambda c: c.densities[1],
    "p": lambda c: (c.densities[0] + c.densities[1]) * CS2,
    "ux": lambda c: c.velocity[0],
    "uy": lambda c: c.velocity[1],
    "uz": lambda c: c.velocity[2],
    "umag": lambda c: (c.velocity[0] ** 2 + c.velocity[1] ** 2 + c.velocity[2] ** 2) ** 0.5,
    "u": lambda c: np.moveaxis(np.asarray(c.velocity), 0, -1),  # (3,nz,ny,nx) -> (nz,ny,nx,3)
}


def derive(name, state, ctx=None):
    """Host NumPy array for field ``name`` from ``state`` (optionally a shared _Ctx)."""
    if name not in FIELDS:
        raise KeyError(f"unknown field {name!r}; valid: {sorted(FIELDS)}")
    ctx = ctx or _Ctx(state)
    return np.asarray(FIELDS[name](ctx))


import h5py


def _scalar_chunks(shape):
    _nz, ny, nx = shape
    return (1, ny, nx)


def _build_xdmf(h5name, shape, steps, fields, has_rock, dtype):
    nz, ny, nx = shape
    prec = np.dtype(dtype).itemsize
    out = [
        '<?xml version="1.0" ?>',
        '<Xdmf Version="2.0">',
        "<Domain>",
        '<Grid Name="series" GridType="Collection" CollectionType="Temporal">',
    ]
    for step in steps:
        g = f"steps/{step:09d}"
        out += [
            f'<Grid Name="t{step}" GridType="Uniform">',
            f'<Time Value="{step}"/>',
            f'<Topology TopologyType="3DCoRectMesh" Dimensions="{nz} {ny} {nx}"/>',
            '<Geometry GeometryType="ORIGIN_DXDYDZ">',
            '<DataItem Dimensions="3" Format="XML">0 0 0</DataItem>',
            '<DataItem Dimensions="3" Format="XML">1 1 1</DataItem>',
            "</Geometry>",
        ]
        for fld in fields:
            if fld == "u":
                out += [
                    '<Attribute Name="u" AttributeType="Vector" Center="Node">',
                    (f'<DataItem Dimensions="{nz} {ny} {nx} 3" NumberType="Float" '
                    f'Precision="{prec}" Format="HDF">{h5name}:/{g}/u</DataItem>'),
                    "</Attribute>",
                ]
            else:
                out += [
                    f'<Attribute Name="{fld}" AttributeType="Scalar" Center="Node">',
                    (f'<DataItem Dimensions="{nz} {ny} {nx}" NumberType="Float" '
                    f'Precision="{prec}" Format="HDF">{h5name}:/{g}/{fld}</DataItem>'),
                    "</Attribute>",
                ]
        if has_rock:
            out += [
                '<Attribute Name="rock" AttributeType="Scalar" Center="Node">',
                (f'<DataItem Dimensions="{nz} {ny} {nx}" NumberType="UChar" '
                f'Precision="1" Format="HDF">{h5name}:/rock</DataItem>'),
                "</Attribute>",
            ]
        out.append("</Grid>")
    out += ["</Grid>", "</Domain>", "</Xdmf>"]
    return "\n".join(out)


class FieldWriter:
    """Append derived fields for a whole run into one HDF5 file + XDMF sidecar."""

    def __init__(
        self,
        path,
        solid=None,
        *,
        fields=("phi", "p"),
        dtype=np.float32,
        compression="gzip",
        clevel=4,
        mask_solid=False,
        attrs=None,
        meta=None,
    ):
        self.path = Path(path)
        self.xdmf_path = self.path.with_suffix(".xdmf")
        self.fields = list(fields)
        self.dtype = np.dtype(dtype)
        self.compression = compression
        self.clevel = clevel
        self.mask_solid = mask_solid
        self._solid = None if solid is None else np.asarray(solid).astype(bool)
        self._extra = dict(attrs or {})
        self._meta = meta
        self.shape = None
        self._f = h5py.File(self.path, "a")

        try:
            if "shape" in self._f.attrs:  # reopening an existing series
                if self._f.attrs["fields"] != ",".join(self.fields):
                    raise ValueError(f"fields mismatch: file has {self._f.attrs['fields']!r}, asked {self.fields}")
                if self._f.attrs["dtype"] != self.dtype.name:
                    raise ValueError(f"dtype mismatch: file has {self._f.attrs['dtype']}, asked {self.dtype.name}")
                self.shape = tuple(int(x) for x in self._f.attrs["shape"])
                if self._solid is not None and self._solid.shape != self.shape:
                    raise ValueError(f"solid shape {self._solid.shape} != file shape {self.shape}")
            elif self._solid is not None:
                self._init_meta(self._solid.shape)

            if self._solid is not None and "rock" not in self._f:
                self._f.create_dataset(
                    "rock",
                    data=self._solid.astype(np.uint8),
                    chunks=_scalar_chunks(self._solid.shape),
                    compression=self.compression,
                    compression_opts=self.clevel,
                )
            if meta is not None:  # mirror the static sections: the HDF5 stays self-describing
                self._f.attrs["run_meta"] = meta.static_json()
        except Exception:
            self._f.close()
            raise

    def _init_meta(self, shape):
        self.shape = tuple(int(x) for x in shape)
        self._f.attrs["shape"] = np.array(self.shape, dtype="int64")
        self._f.attrs["fields"] = ",".join(self.fields)
        self._f.attrs["dtype"] = self.dtype.name
        self._f.attrs["cs2"] = float(CS2)
        for k, v in self._extra.items():
            self._f.attrs[k] = v

    def append(self, step, state):
        ctx = _Ctx(state)
        if self.shape is None:
            self._init_meta(np.asarray(ctx.densities[0]).shape)
        grp = self._f.require_group(f"steps/{step:09d}")
        for name in self.fields:
            arr = np.asarray(FIELDS[name](ctx))
            if arr.dtype.kind == "f":
                arr = arr.astype(self.dtype)
            if self.mask_solid and self._solid is not None and arr.dtype.kind == "f":
                arr[self._solid] = np.nan  # broadcasts over the trailing 3 for vector u
            if name in grp:
                del grp[name]
            chunks = (1, *self.shape[1:], 3) if name == "u" else _scalar_chunks(self.shape)
            grp.create_dataset(name, data=arr, chunks=chunks, compression=self.compression, compression_opts=self.clevel)
        self._f.flush()
        self._write_xdmf()
        if self._meta is not None:
            self._meta.update(step)

    def _write_xdmf(self):
        steps = sorted(int(k) for k in self._f["steps"]) if "steps" in self._f else []
        xml = _build_xdmf(self.path.name, self.shape, steps, self.fields, "rock" in self._f, self.dtype)
        self.xdmf_path.write_text(xml, encoding="utf-8")

    def close(self):
        if self._f:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FieldReader:
    """Load only the requested field / step / slice from a FieldWriter file."""

    def __init__(self, path):
        self.path = Path(path)
        self._f = h5py.File(self.path, "r")

    @property
    def fields(self):
        return list(self._f.attrs["fields"].split(","))

    @property
    def steps(self):
        return sorted(int(k) for k in self._f["steps"]) if "steps" in self._f else []

    @property
    def shape(self):
        return tuple(int(x) for x in self._f.attrs["shape"])

    @property
    def attrs(self):
        return {k: self._f.attrs[k] for k in self._f.attrs}

    @property
    def rock(self):
        if "rock" not in self._f:
            raise KeyError("no rock mask in this file")
        return self._f["rock"][()].astype(bool)

    def read(self, name, step=None, sl=None):
        if name not in self.fields:
            raise KeyError(f"no field {name!r}; present: {self.fields}")
        steps = self.steps
        if not steps:
            raise ValueError("file has no steps")
        step = steps[-1] if step is None else step
        if step not in steps:
            raise ValueError(f"no step {step}; present: {steps}")
        ds = self._f[f"steps/{step:09d}/{name}"]
        return ds[()] if sl is None else ds[sl]

    def close(self):
        if self._f:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_fields(path, names, step=None):
    """Dict of {name: array} read selectively from ``path`` at ``step`` (default latest)."""
    with FieldReader(path) as r:
        return {n: r.read(n, step=step) for n in names}


def _git_commit():
    """HEAD sha of the bob checkout (best-effort; None outside git)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return r.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _local_now():
    """Timezone-aware local time (default ``now`` for RunMeta)."""
    return datetime.now(timezone.utc).astimezone()


def _jsonable(o):
    return o.item() if hasattr(o, "item") else str(o)


class RunMeta:
    """Live run-metadata sidecar: an atomically rewritten ``run_meta.json``.

    Static sections (solver ``Params``, geometry stats, environment) are
    captured once at construction; ``update(step)`` is the heartbeat that
    refreshes the ``progress`` section (steps/s, ETA, elapsed and device
    hours) — hook it to the run's block cadence (``FieldWriter(meta=)`` /
    ``RunMonitor(meta=)`` do this automatically). An existing file at ``path``
    is treated as an earlier segment of the same run: its elapsed/device hours
    carry forward (cf. ``stats.carry_rows``), and with ``resume_step > 0`` the
    original start time, the ``extra`` section and a ``run.resumes`` history of
    segment boundaries are kept (see ``_carry_resume``). A run that dies leaves
    ``status: "running"`` with a stale ``updated_at`` — that IS the
    interrupted signal; call ``finish(step)`` on normal completion.
    ``clock``/``now`` are injectable for tests.
    """

    def __init__(
        self,
        path,
        *,
        params=None,
        solid=None,
        geometry_source=None,
        target_steps=None,
        lattice=None,
        extra=None,
        notes=None,
        resume_step=0,
        clock=time.monotonic,
        now=_local_now,
    ):
        self.path = Path(path)
        self._clock, self._now = clock, now
        self._t0 = clock()
        self._step0 = int(resume_step)
        self.target_steps = None if target_steps is None else int(target_steps)
        self._prev_hours = (0.0, 0.0)  # (elapsed, device) carried from earlier segments
        prev_doc = None
        if self.path.exists():
            # unreadable/partial earlier file is non-fatal: start the hour ledger fresh
            with contextlib.suppress(OSError, ValueError, KeyError, TypeError):
                doc = json.loads(self.path.read_text())
                prev = doc["progress"]
                self._prev_hours = (float(prev["elapsed_hours"]), float(prev["device_hours"]))
                prev_doc = doc
        mask = None if solid is None else np.asarray(solid).astype(bool)
        self._doc = {
            "run": {
                "start_time": now().isoformat(timespec="seconds"),
                "end_time": None,
                "status": "running",
                "argv": list(sys.argv),
                "cwd": os.getcwd(),
                "resume_step": self._step0,
                "notes": notes,
            },
            "solver": self._solver(params, mask, lattice),
            "geometry": self._geometry(mask, geometry_source),
            "environment": self._environment(),
            "progress": {},
            "extra": dict(extra or {}),
        }
        if prev_doc is not None and self._step0 > 0:
            self._carry_resume(prev_doc)
        self.update(self._step0)

    def _carry_resume(self, prev):
        """Resume bookkeeping: keep the run's original ``start_time`` (the resume time goes to
        ``resumed_time``), append a ``run.resumes`` entry recording how the previous segment
        ended, and carry ``extra`` forward. The previous ``finish_type``/``finish_detail`` move
        into that entry — they describe a finished segment, not the live run — and a ``None``
        in the new ``extra`` does not overwrite a carried value."""
        run, prun = self._doc["run"], prev.get("run") or {}
        carried = dict(prev.get("extra") or {})
        run["resumed_time"] = run["start_time"]
        run["start_time"] = prun.get("start_time") or run["start_time"]
        entry = {
            "at": run["resumed_time"],
            "resume_step": self._step0,
            "argv": list(sys.argv),
            "prev_status": prun.get("status"),
            "prev_finish_type": carried.pop("finish_type", None),
            "prev_finish_detail": carried.pop("finish_detail", None),
            "prev_step": (prev.get("progress") or {}).get("step"),
            "prev_argv": prun.get("argv"),
        }
        run["resumes"] = [*(prun.get("resumes") or []), entry]
        for k, v in self._doc["extra"].items():
            if v is not None or k not in carried:
                carried[k] = v
        self._doc["extra"] = carried

    @staticmethod
    def _solver(params, mask, lattice):
        sec = {} if params is None else {k: (list(v) if isinstance(v, tuple) else v) for k, v in params._asdict().items()}
        from bob import d3q19

        sec["precision"] = str(d3q19.W.dtype)  # actual lattice dtype, not env vars
        sec["lattice"] = lattice if lattice is not None else (None if mask is None else {3: "d3q19"}.get(mask.ndim))
        sec["bob_commit"] = _git_commit()
        return sec

    @staticmethod
    def _geometry(mask, source):
        if mask is None:
            return None
        frac = float(mask.mean())
        return {
            "shape": list(mask.shape),
            "solid_fraction": frac,
            "porosity": 1.0 - frac,
            "sha256": hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
            "source": source,
        }

    @staticmethod
    def _environment():
        import jax

        devs = jax.devices()
        return {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "jax_version": jax.__version__,
            "backend": jax.default_backend(),
            "devices": [d.device_kind for d in devs],
            "n_devices": len(devs),
            "cpu_count": os.cpu_count(),
        }

    def update(self, step):
        """Heartbeat: refresh the progress section and rewrite the file."""
        step = int(step)
        dt = self._clock() - self._t0  # seconds, this segment only
        dstep = step - self._step0
        rate = dstep / dt if dt > 0 and dstep > 0 else None
        n_dev = self._doc["environment"]["n_devices"]
        eta = None
        if rate and self.target_steps is not None and step < self.target_steps:
            eta = (self._now() + timedelta(seconds=(self.target_steps - step) / rate)).isoformat(timespec="seconds")
        self._doc["progress"] = {
            "step": step,
            "target_steps": self.target_steps,
            "fraction": None if not self.target_steps else step / self.target_steps,
            "steps_per_s": rate,
            "elapsed_hours": self._prev_hours[0] + dt / 3600.0,
            "device_hours": self._prev_hours[1] + dt / 3600.0 * n_dev,
            "eta": eta,
            "updated_at": self._now().isoformat(timespec="seconds"),
        }
        self._write()

    def finish(self, step, **extra):
        """Mark the run complete: final progress + status/end_time. Keyword
        arguments (e.g. ``finish_type="pv_cap"``) are merged into the ``extra``
        section — how a run records *why* it ended."""
        self.update(step)
        self._doc["run"]["status"] = "finished"
        self._doc["run"]["end_time"] = self._now().isoformat(timespec="seconds")
        if extra:
            self._doc["extra"].update(extra)
        self._write()

    def static_json(self):
        """Everything except progress, as a JSON string (for HDF5 mirroring)."""
        return json.dumps({k: v for k, v in self._doc.items() if k != "progress"}, default=_jsonable)

    def _write(self):
        tmp = self.path.with_name(self.path.name + ".tmp")  # same dir => atomic os.replace
        tmp.write_text(json.dumps(self._doc, indent=2, default=_jsonable), encoding="utf-8")
        os.replace(tmp, self.path)
