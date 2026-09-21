"""Tests for bob.utils.file field I/O: HDF5 FieldWriter/FieldReader round-trips, derived fields, XDMF sidecar."""

import h5py
import jax.numpy as jnp
import numpy as np
import pytest

from bob import color3d
from bob.color3d import color_field
from bob.d3q19 import CS2
from bob.utils import file as bobfile


def test_writer_creates_groups_attrs_and_rock(tmp_path):
    s = tiny_state()
    solid = np.zeros((4, 5, 6), bool)
    solid[0, 0, 0] = True
    p = tmp_path / "run.h5"
    with bobfile.FieldWriter(p, solid, fields=("phi", "p"), attrs={"theta": 140.0}) as w:
        w.append(2000, s)
        w.append(4000, s)
    with h5py.File(p, "r") as f:
        assert tuple(f.attrs["shape"]) == (4, 5, 6)
        assert f.attrs["fields"] == "phi,p"
        assert f.attrs["dtype"] == "float32"
        assert np.isclose(f.attrs["cs2"], 1.0 / 3.0)
        assert np.isclose(f.attrs["theta"], 140.0)
        assert f["rock"].dtype == np.uint8 and bool(f["rock"][0, 0, 0]) is True
        assert sorted(f["steps"].keys()) == ["000002000", "000004000"]
        phi = f["steps/000002000/phi"]
        assert phi.dtype == np.float32 and phi.shape == (4, 5, 6)
        assert phi.chunks is not None and phi.compression == "gzip"
        np.testing.assert_allclose(phi[()], bobfile.derive("phi", s).astype(np.float32))


def test_writer_resume_appends_and_keeps_rock(tmp_path):
    s = tiny_state()
    solid = np.zeros((4, 5, 6), bool)
    p = tmp_path / "run.h5"
    with bobfile.FieldWriter(p, solid, fields=("phi",)) as w:
        w.append(1000, s)
    with bobfile.FieldWriter(p, solid, fields=("phi",)) as w:  # reopen
        w.append(2000, s)
    with h5py.File(p, "r") as f:
        assert sorted(f["steps"].keys()) == ["000001000", "000002000"]
        assert "rock" in f


def test_writer_schema_mismatch_raises(tmp_path):
    s = tiny_state()
    solid = np.zeros((4, 5, 6), bool)
    p = tmp_path / "run.h5"
    with bobfile.FieldWriter(p, solid, fields=("phi",)) as w:
        w.append(1000, s)
    with pytest.raises(ValueError):
        bobfile.FieldWriter(p, solid, fields=("phi", "p"))  # different field set
    with h5py.File(p, "a") as f2:  # the failed constructor must not have leaked the handle
        assert "rock" in f2


def test_writer_mask_solid_sets_nan(tmp_path):
    s = tiny_state()
    solid = np.zeros((4, 5, 6), bool)
    solid[1, 2, 3] = True
    p = tmp_path / "run.h5"
    with bobfile.FieldWriter(p, solid, fields=("phi",), mask_solid=True) as w:
        w.append(1000, s)
    with h5py.File(p, "r") as f:
        assert np.isnan(f["steps/000001000/phi"][1, 2, 3])


def tiny_state(nz=4, ny=5, nx=6):
    """A deterministic, non-uniform State for round-trip tests (no RNG)."""
    base = jnp.asarray(np.linspace(0.02, 0.2, 19 * nz * ny * nx).reshape(19, nz, ny, nx))
    fR = base
    fB = base[::-1] * 0.5  # break R/B symmetry so phi/velocity are non-trivial
    return color3d.State(fR, fB)


def test_reader_roundtrip_steps_and_partial(tmp_path):
    s = tiny_state()
    solid = np.zeros((4, 5, 6), bool)
    solid[2, 3, 4] = True
    p = tmp_path / "run.h5"
    with bobfile.FieldWriter(p, solid, fields=("phi", "ux")) as w:
        w.append(2000, s)
        w.append(4000, s)

    r = bobfile.FieldReader(p)
    assert r.fields == ["phi", "ux"]
    assert r.steps == [2000, 4000]
    assert r.shape == (4, 5, 6)
    assert r.rock.dtype == bool and r.rock[2, 3, 4]

    full = r.read("phi", step=2000)
    np.testing.assert_allclose(full, bobfile.derive("phi", s).astype(np.float32))
    # default step is the latest
    np.testing.assert_allclose(r.read("phi"), r.read("phi", step=4000))
    # partial slice equals the full field indexed
    np.testing.assert_allclose(r.read("phi", step=2000, sl=np.s_[2]), full[2])


def test_reader_missing_raises(tmp_path):
    s = tiny_state()
    solid = np.zeros((4, 5, 6), bool)
    p = tmp_path / "run.h5"
    with bobfile.FieldWriter(p, solid, fields=("phi",)) as w:
        w.append(1000, s)
    r = bobfile.FieldReader(p)
    with pytest.raises(KeyError):
        r.read("ux")
    with pytest.raises(ValueError):
        r.read("phi", step=999999)


def test_read_fields_convenience(tmp_path):
    s = tiny_state()
    solid = np.zeros((4, 5, 6), bool)
    p = tmp_path / "run.h5"
    with bobfile.FieldWriter(p, solid, fields=("phi", "ux")) as w:
        w.append(1000, s)
    d = bobfile.read_fields(p, ("phi", "ux"), step=1000)
    assert set(d) == {"phi", "ux"}
    np.testing.assert_allclose(d["phi"], bobfile.derive("phi", s).astype(np.float32))


def test_derive_matches_direct_computation():
    s = tiny_state()
    rhoR, rhoB = color3d.densities(s)
    u = color3d.velocity(s)
    np.testing.assert_allclose(bobfile.derive("phi", s), np.asarray(color_field(rhoR, rhoB)))
    np.testing.assert_allclose(bobfile.derive("rho", s), np.asarray(rhoR + rhoB))
    np.testing.assert_allclose(bobfile.derive("p", s), np.asarray((rhoR + rhoB) * CS2))
    np.testing.assert_allclose(bobfile.derive("ux", s), np.asarray(u[0]))
    np.testing.assert_allclose(bobfile.derive("uz", s), np.asarray(u[2]))
    assert bobfile.derive("u", s).shape == (4, 5, 6, 3)  # (nz,ny,nx,3) on-disk order


def test_derive_unknown_field_raises():
    s = tiny_state()
    with pytest.raises(KeyError):
        bobfile.derive("nope", s)


def test_xdmf_is_temporal_collection_referencing_datasets(tmp_path):
    import xml.etree.ElementTree as ET

    s = tiny_state()
    solid = np.zeros((4, 5, 6), bool)
    p = tmp_path / "run.h5"
    with bobfile.FieldWriter(p, solid, fields=("phi", "u")) as w:
        w.append(2000, s)
        w.append(4000, s)
    xdmf = (tmp_path / "run.xdmf").read_text()
    root = ET.fromstring(xdmf)  # raises if malformed
    coll = root.find(".//Grid[@CollectionType='Temporal']")
    assert coll is not None
    grids = coll.findall("Grid")
    assert len(grids) == 2
    assert grids[0].find("Time").get("Value") == "2000"
    # the vector u is declared as a Vector attribute with trailing-3 dims
    u_attr = next(a for g in grids for a in g.findall("Attribute") if a.get("Name") == "u")
    assert u_attr.get("AttributeType") == "Vector"
    assert u_attr.find("DataItem").get("Dimensions").split()[-1] == "3"
    # every DataItem that points into the h5 references the real file + an existing path
    with h5py.File(p, "r") as f:
        for di in root.iter("DataItem"):
            if di.get("Format") == "HDF":
                fname, dpath = di.text.strip().split(":")
                assert fname == "run.h5"
                assert dpath.lstrip("/") in f or dpath.lstrip("/").split("/")[0] in f
