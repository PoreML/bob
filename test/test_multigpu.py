"""Multi-device parity tests: sharded runs must reproduce single-device results.

Each test runs ``multigpu_parity.py`` in a subprocess with
``XLA_FLAGS=--xla_force_host_platform_device_count=4`` + ``JAX_PLATFORMS=cpu``:
the device-count flag must land before jax initializes (this pytest process
already initialized it via conftest), and CPU emulation makes the tests run
identically on laptops, CI, and GPU nodes. The harness asserts sharded ==
single-device to 1e-12 on 2- and 4-device meshes along EVERY shard axis
(z, y, x); see its docstring for what each case covers.
"""

import os
import subprocess
import sys
from pathlib import Path


def _run(case):
    env = os.environ.copy()
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4").strip()
    env["JAX_PLATFORMS"] = "cpu"
    script = Path(__file__).with_name("multigpu_parity.py")
    proc = subprocess.run([sys.executable, str(script), case], capture_output=True, text=True, env=env, timeout=600, check=False)
    assert proc.returncode == 0, f"parity case {case!r} failed:\n{proc.stdout}\n{proc.stderr}"


def test_single_phase_parity():
    _run("single")


def test_drainage_clamp_wetting_parity():
    _run("clamp")


def test_drainage_piston_parity():
    _run("piston")


def test_drainage_velocity_inlet_parity():
    _run("velocity")


def test_wetting_sealed_zwalls_parity():
    _run("wetting")


def test_zou_he_washburn_parity():
    _run("zouhe")


def test_capillary_pressure_layers_parity():
    _run("pc")


def test_halo_step_clamp_parity():
    _run("halo_clamp")


def test_halo_step_piston_parity():
    _run("halo_piston")


def test_halo_step_velocity_parity():
    _run("halo_velocity")


def test_halo_step_depth_is_minimal():
    _run("halo_depth")


def test_staged_halo_clamp_parity():
    _run("staged_clamp")


def test_staged_halo_piston_parity():
    _run("staged_piston")


def test_staged_halo_velocity_parity():
    _run("staged_velocity")


def test_staged_halo_theta_field_parity():
    _run("staged_theta")


def test_staged_phi_halo_depth_is_minimal():
    _run("staged_depth")
