"""Optional analysis curves: a run writes data and field renders by default, and the
curve plots only when plots are enabled (``--plots`` on a driver, or BOB_PLOTS=1).

The gate lives in ``bob.utils.viz``; ``RunMonitor`` is the shared consumer that would
otherwise write a progress curve next to every metrics.csv.
"""

from __future__ import annotations

import importlib

import pytest

from bob.utils import stats, viz


@pytest.fixture(autouse=True)
def restore_gate():
    """Each test starts from plots-off and leaves the process state untouched."""
    before = viz.plots_enabled()
    viz.enable_plots(False)
    yield
    viz.enable_plots(before)


def test_plots_are_off_by_default(monkeypatch):
    monkeypatch.delenv("BOB_PLOTS", raising=False)
    assert importlib.reload(viz).plots_enabled() is False


def test_env_var_enables_plots(monkeypatch):
    monkeypatch.setenv("BOB_PLOTS", "1")
    assert importlib.reload(viz).plots_enabled() is True
    monkeypatch.delenv("BOB_PLOTS")
    importlib.reload(viz)  # leave the module in its default state for later tests


def test_enable_plots_toggles(tmp_path):
    viz.enable_plots(True)
    assert viz.plots_enabled() is True
    viz.enable_plots(False)
    assert viz.plots_enabled() is False


def test_line_plot_writes_nothing_when_disabled(tmp_path):
    path = tmp_path / "curve.svg"
    viz.line_plot(path, [([0, 1], [0.0, 1.0], "s", {})], "t", "x", "y")
    assert not path.exists()


def test_line_plot_writes_when_enabled(tmp_path):
    viz.enable_plots(True)
    path = tmp_path / "curve.svg"
    viz.line_plot(path, [([0, 1], [0.0, 1.0], "s", {})], "t", "x", "y")
    assert path.exists() and path.stat().st_size > 0


def test_run_monitor_logs_metrics_without_the_curve(tmp_path):
    """The data a run is judged on is never gated: metrics.csv is always written."""
    mon = stats.RunMonitor(tmp_path, curve_field="saturation")
    mon.log(step=100, saturation=0.25, front=3.0)
    assert (tmp_path / "metrics.csv").exists()
    assert not (tmp_path / "saturation.svg").exists()


def test_run_monitor_writes_the_curve_when_enabled(tmp_path):
    viz.enable_plots(True)
    mon = stats.RunMonitor(tmp_path, curve_field="saturation")
    mon.log(step=100, saturation=0.25, front=3.0)
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "saturation.svg").exists()
