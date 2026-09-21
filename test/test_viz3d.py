"""viz3d render robustness — the EGL-after-CUDA black-frame escalation.

On headless GPU nodes (no X) VTK falls back to vtkEGLRenderWindow; an EGL
context created right after CUDA activity on the same GPU stalls 20-27 s and
screenshots all black (a driver interop quirk). A real frame is never
all-black (light background), so on the first black shot _render escalates
to plotting in a fresh subprocess (clean EGL context: fast + valid), with one
in-process retry as the last resort. Logic tests fake the screenshot/subprocess layers — no GL needed; the
worker round-trip test renders for real.
"""

import numpy as np
import pytest
import pyvista as pv
from PIL import Image

from bob.utils import viz3d

BLACK = np.zeros((48, 64, 3), np.uint8)
GOOD = np.full((48, 64, 3), 200, np.uint8)
RHON = np.full((4, 4, 4), 1.0)


@pytest.fixture(autouse=True)
def reset_escalation():
    viz3d._use_subprocess = False
    yield
    viz3d._use_subprocess = False


def _fake_screenshots(monkeypatch, images):
    """Make successive Plotter.screenshot calls return ``images`` in order."""
    calls = []

    def fake(self, *args, **kwargs):
        img = images[min(len(calls), len(images) - 1)]
        calls.append(1)
        return img

    monkeypatch.setattr(pv.Plotter, "screenshot", fake)
    return calls


def _fake_subprocess(monkeypatch, result=None):
    calls = []

    def fake(rhoN, solid, kwargs):
        calls.append(1)
        if result is None:
            raise OSError("no subprocess in this test")
        return result

    monkeypatch.setattr(viz3d, "_render_subprocess", fake)
    return calls


def test_good_screenshot_stays_in_process(monkeypatch, tmp_path):
    shots = _fake_screenshots(monkeypatch, [GOOD])
    subs = _fake_subprocess(monkeypatch, GOOD)
    viz3d.phase_frame(tmp_path / "f.png", RHON, None)
    assert len(shots) == 1 and len(subs) == 0
    assert not viz3d._use_subprocess


def test_black_screenshot_escalates_to_subprocess(monkeypatch, tmp_path):
    shots = _fake_screenshots(monkeypatch, [BLACK])
    subs = _fake_subprocess(monkeypatch, GOOD)
    viz3d.phase_frame(tmp_path / "f.png", RHON, None)
    assert len(shots) == 1 and len(subs) == 1
    assert np.asarray(Image.open(tmp_path / "f.png")).max() > 0
    # escalation is sticky: the next frame goes straight to the subprocess
    viz3d.phase_frame(tmp_path / "g.png", RHON, None)
    assert len(shots) == 1 and len(subs) == 2


def test_subprocess_failure_falls_back_to_inprocess_retry(monkeypatch, tmp_path):
    shots = _fake_screenshots(monkeypatch, [BLACK, GOOD])
    subs = _fake_subprocess(monkeypatch, None)  # raises
    viz3d.phase_frame(tmp_path / "f.png", RHON, None)
    assert len(subs) == 1 and len(shots) == 2
    assert np.asarray(Image.open(tmp_path / "f.png")).max() > 0
    assert not viz3d._use_subprocess  # broken subprocess must not stick


def test_all_black_raises_instead_of_saving(monkeypatch, tmp_path):
    """A transiently poisoned GPU (e.g. another process's CUDA) can black every
    attempt — that must surface as an error, never a silently saved black PNG."""
    shots = _fake_screenshots(monkeypatch, [BLACK])
    subs = _fake_subprocess(monkeypatch, BLACK)  # worker black -> internal failure
    with pytest.raises(RuntimeError, match="black"):
        viz3d.phase_frame(tmp_path / "f.png", RHON, None)
    assert len(shots) >= 2 and len(subs) >= 1  # it did exhaust both strategies
    assert not (tmp_path / "f.png").exists()


def test_escalated_worker_failure_recovers_in_process(monkeypatch, tmp_path):
    viz3d._use_subprocess = True
    shots = _fake_screenshots(monkeypatch, [GOOD])
    subs = _fake_subprocess(monkeypatch, None)  # worker broken
    viz3d.phase_frame(tmp_path / "f.png", RHON, None)
    assert len(subs) == 1 and len(shots) == 1
    assert not viz3d._use_subprocess  # local success unsticks the escalation


def test_worker_roundtrip(tmp_path):
    """Real subprocess render: valid non-black frame with rock + red phase."""
    z, y, x = np.mgrid[0:16, 0:16, 0:16]
    rhoN = np.where((z - 8) ** 2 + (y - 8) ** 2 + (x - 8) ** 2 < 5**2, 1.0, -1.0)
    solid = np.zeros(rhoN.shape, dtype=bool)
    solid[:, :, :2] = True
    shot = viz3d._render_subprocess(
        rhoN, solid,
        {"window": (320, 240), "rock_opacity": 0.12, "text": "t", "camera": "iso", "smooth": False,
         "zoom": 1.3, "shade": False, "font_size": 11},
    )
    assert shot.shape[2] == 3 and shot.any()
