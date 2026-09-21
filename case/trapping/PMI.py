# ---------------------------------------------------------------------------
# Vendored from https://github.com/Omid-Tavakkoli/PMI (PMI.py), unmodified body.
# Pore-Morphology-based Initializer. No upstream LICENSE file exists; vendored
# here for research use only. Cite:
#   Tavakkoli, Wang, Mostaghimi, Armstrong (2025), "A pore morphology-based
#   initializer to accelerate lattice Boltzmann simulations of capillary-
#   dominated flow with variable wettability", Physics of Fluids 37(9),
#   doi:10.1063/5.0285656.
# trapping_demo.py imports its functions (compute_saturation_parallel,
# find_best_kernel_size); the __main__ / input.txt path is not used.
# Do not lint, format or edit the body below.
# ---------------------------------------------------------------------------
import gc
import math
import multiprocessing as mp
import platform
import time
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import skimage as ski

# --------------------------------------------------------------------------------------
# Caching utilities & multi-pass morphology to handle large radii kernel efficiently
# --------------------------------------------------------------------------------------

MAX_SE_RADIUS = 10  # radii above this threshold are decomposed into multi-pass operations
_SE_CACHE: dict[int, np.ndarray] = {}

# Global domain reference used only on Unix-like systems with 'fork' start method
FORK_DOMAIN: np.ndarray | None = None

def _get_ball(radius: int) -> np.ndarray:
    """Return a cached spherical structuring element of the given *radius*."""
    if radius <= 0:
        raise ValueError("Structuring-element radius must be positive")
    ball = _SE_CACHE.get(radius)
    if ball is None:
        ball = ski.morphology.ball(radius, dtype=np.uint8)
        _SE_CACHE[radius] = ball
    return ball

def _morphology_multi_pass(volume: np.ndarray, total_radius: int, op: str) -> np.ndarray:
    """Apply binary *op* ('dilation' or 'erosion') with possibly large *total_radius* by
    chaining multiple passes whose individual radii do not exceed *MAX_SE_RADIUS*.
    """
    if total_radius <= 0:
        return volume

    remaining = total_radius
    result = volume
    while remaining > 0:
        step = min(remaining, MAX_SE_RADIUS)
        se = _get_ball(step)
        if op == "dilation":
            result = ski.morphology.dilation(result, footprint=se).astype(bool)
        elif op == "erosion":
            result = ski.morphology.erosion(result, footprint=se).astype(bool)
        else:
            raise ValueError(f"Unsupported morphology operation: {op}")
        remaining -= step
    return result

# --------------------------------------------------------------------------------------
# Saturation computation rules
# --------------------------------------------------------------------------------------

def compute_saturation(domain: np.ndarray, k: int, theta: float) -> tuple[float, np.ndarray]:
    """Compute saturation and combined phase array for *domain* (single-process)."""
    kernel_size_solid = max(round(k * math.cos(math.radians(theta))), 1)
    kernel_size_nwp = k

    # Use cached, multi-pass morphology for large radii
    grain_dilation = _morphology_multi_pass(domain, kernel_size_solid, "dilation")
    nwp_dilation   = _morphology_multi_pass(grain_dilation, kernel_size_nwp, "erosion")

    comb = np.copy(domain)
    comb[comb == 0] = 3
    comb[comb == 1] = 0
    comb[comb == 3] = 2
    comb[nwp_dilation == 0] = 1
    comb[domain == 1] = 0

    wp  = np.sum(comb == 2)
    nwp = np.sum(comb == 1)
    return wp / (wp + nwp), comb

# --------------------------------------------------------------------------------------
# Step size determination
# --------------------------------------------------------------------------------------

def _determine_step_size(gap: float) -> int:
    """Return kernel size increment based on the magnitude of *gap* (target − current)."""
    gap = abs(gap)
    if gap > 0.95:
        return 20
    elif gap > 0.8:
        return 18
    elif gap > 0.6:
        return 16
    elif gap > 0.5:
        return 14
    elif gap > 0.4:
        return 12
    elif gap > 0.3:
        return 10
    elif gap > 0.2:
        return 8
    elif gap > 0.1:
        return 6
    elif gap > 0.05:
        return 4
    elif gap > 0.01:
        return 2
    else:
        return 1

# --------------------------------------------------------------------------------------
# Finding the best kernel size for a given saturation to be simulated on full domain
# --------------------------------------------------------------------------------------

def find_best_kernel_size(domain_test: np.ndarray, target_sat: float, theta: float, num_workers: int, tol: float = 1e-6, max_kernel_size: int = 100):
    """Find kernel size that matches *target_sat* on *domain_test* within *tol*"""
    kernel_size = 1
    best_kernel_size = kernel_size
    best_diff = float("inf")
    tested_sizes = set()

    while kernel_size <= max_kernel_size:
        saturation, _ = compute_saturation_parallel(domain_test, kernel_size, theta, num_workers)
        tested_sizes.add(kernel_size)
        diff = abs(saturation - target_sat)
        print(f"Kernel size {kernel_size} => Saturation: {saturation:.4f} (diff: {diff:.4f})")

        if diff < tol:
            best_kernel_size = kernel_size
            best_diff = 0
            break

        if diff < best_diff:
            best_diff = diff
            best_kernel_size = kernel_size

        if saturation >= target_sat:
            # stop only after two consecutive increases
            consecutive_worse = 0
            prev_diff = diff  # start with the diff of the current (kernel_size) value
            for k in range(kernel_size - 1, 0, -1):
                if k in tested_sizes:
                    continue
                sat_k, _ = compute_saturation_parallel(domain_test, k, theta, num_workers)
                tested_sizes.add(k)
                diff_k = abs(sat_k - target_sat)
                print(f"Downward search - Kernel size {k} => Saturation: {sat_k:.4f} (diff: {diff_k:.4f})")

                # Update global best if this k is better
                if diff_k < best_diff:
                    best_diff = diff_k
                    best_kernel_size = k

                # Check monotonicity: we stop only if diff increases twice *in a row*
                if diff_k > prev_diff:
                    consecutive_worse += 1
                    if consecutive_worse >= 2:
                        print("Stopping downward search")
                        break
                else:
                    consecutive_worse = 0  # reset if improvement or equal

                prev_diff = diff_k  # update reference for next iteration
            # -----------------------------------------------------------------------
            break

        gap = target_sat - saturation
        step = _determine_step_size(gap)

        kernel_size += step

    return best_kernel_size, best_diff

# --------------------------------------------------------------------------------------
# Parallel helpers
# --------------------------------------------------------------------------------------

def _process_chunk(args):
    """Worker function executed in a separate process.

    Parameters received via a picklable tuple to be compatible with Windows spawn:
    (shm_name, shape, dtype_str, k, theta, z_start, z_end)
    """
    shm_name, shape, dtype_str, k, theta, z_start, z_end = args

    # Attach to the shared memory block containing the domain volume
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        domain = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        # Determine halo padding – must cover influence of both morphology steps
        r1 = max(round(k * math.cos(math.radians(theta))), 1)  # dilation radius
        r2 = round(k)                                          # erosion radius
        halo = r1 + r2                                         # total influence range
        z0 = max(0, z_start - halo)
        z1 = min(domain.shape[0], z_end + halo)

        sub_volume = domain[z0:z1]
        _, comb_sub = compute_saturation(sub_volume, k, theta)

        # Crop off halos so only interior [z_start:z_end] remains
        crop_from = z_start - z0
        crop_to = crop_from + (z_end - z_start)
        comb_crop = comb_sub[crop_from:crop_to]

        # Local counts
        wp_local = np.sum(comb_crop == 2)
        nwp_local = np.sum(comb_crop == 1)
        return comb_crop, wp_local, nwp_local
    finally:
        # Detach from the shared memory segment in the worker
        shm.close()


def _process_chunk_unix(args):
    """Worker function for Unix 'fork' path using a global domain reference.

    Expected args: (None, shape, dtype_str, k, theta, z_start, z_end)
    """
    _, _shape, _dtype_str, k_local, theta_local, z_s, z_e = args
    if FORK_DOMAIN is None:
        raise RuntimeError("FORK_DOMAIN is not initialized in worker")

    domain = FORK_DOMAIN

    r1 = max(round(k_local * math.cos(math.radians(theta_local))), 1)
    r2 = round(k_local)
    halo = r1 + r2
    z0 = max(0, z_s - halo)
    z1 = min(domain.shape[0], z_e + halo)
    sub_volume = domain[z0:z1]
    _, comb_sub = compute_saturation(sub_volume, k_local, theta_local)
    crop_from = z_s - z0
    crop_to = crop_from + (z_e - z_s)
    comb_crop = comb_sub[crop_from:crop_to]
    wp_local = np.sum(comb_crop == 2)
    nwp_local = np.sum(comb_crop == 1)
    return comb_crop, wp_local, nwp_local


def compute_saturation_parallel(domain: np.ndarray, k: int, theta: float, num_workers: int) -> tuple[float, np.ndarray]:
    """Parallel saturation computation using split-volume strategy.

    - On Windows or when using the 'spawn' start method, shared memory is used to avoid
      copying the full volume into each worker.
    - On Unix-like systems with 'fork', the domain is passed directly for zero-copy COW.
    """
    num_workers = max(num_workers, 1)

    z_dim = domain.shape[0]
    chunk_sz = math.ceil(z_dim / num_workers)

    use_shared = (platform.system() == "Windows") or (mp.get_start_method(allow_none=True) == "spawn")

    if use_shared:
        # Create shared memory for the domain to avoid copying under Windows spawn
        shm = shared_memory.SharedMemory(create=True, size=domain.nbytes)
        try:
            domain_shm = np.ndarray(domain.shape, dtype=domain.dtype, buffer=shm.buf)
            domain_shm[...] = domain

            # Prepare tasks with shared memory reference
            tasks = []
            dtype_str = domain.dtype.str
            for i in range(num_workers):
                z_start = i * chunk_sz
                z_end = min(z_dim, (i + 1) * chunk_sz)
                if z_start >= z_dim:
                    break
                tasks.append((shm.name, domain.shape, dtype_str, k, theta, z_start, z_end))

            with mp.Pool(processes=num_workers) as pool:
                results = pool.map(_process_chunk, tasks)

            comb_full = np.empty_like(domain, dtype=np.uint8)
            total_wp = total_nwp = 0
            for (z_start, task), (comb_crop, wp_local, nwp_local) in zip(enumerate(tasks), results):
                z_start = task[5]
                comb_full[z_start : z_start + comb_crop.shape[0]] = comb_crop
                total_wp += wp_local
                total_nwp += nwp_local

            saturation = total_wp / (total_wp + total_nwp)
            return saturation, comb_full
        finally:
            shm.close()
            shm.unlink()
    else:
        # Unix-like with fork: use a global reference to avoid pickling a closure
        global FORK_DOMAIN
        FORK_DOMAIN = domain
        tasks = []
        for i in range(num_workers):
            z_start = i * chunk_sz
            z_end = min(z_dim, (i + 1) * chunk_sz)
            if z_start >= z_dim:
                break
            tasks.append((None, domain.shape, domain.dtype.str, k, theta, z_start, z_end))

        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_process_chunk_unix, tasks)

        comb_full = np.empty_like(domain, dtype=np.uint8)
        total_wp = total_nwp = 0
        for (z_start, task), (comb_crop, wp_local, nwp_local) in zip(enumerate(tasks), results):
            z_start = task[5]
            comb_full[z_start : z_start + comb_crop.shape[0]] = comb_crop
            total_wp += wp_local
            total_nwp += nwp_local

        saturation = total_wp / (total_wp + total_nwp)
        return saturation, comb_full

# --------------------------------------------------------------------------------------
# Modified refinement utilising the parallel saturation routine
# --------------------------------------------------------------------------------------

def refine_kernel_size(domain: np.ndarray, initial_kernel_size: int, initial_saturation: float, initial_diff: float, target_sat: float, theta: float, num_workers: int, tol_full: float = 0.02, max_iterations: int = 20) -> tuple[int, float, np.ndarray]:
    """Refine *kernel size* using parallel morphology for full-domain evaluations."""
    best_kernel_size = initial_kernel_size
    best_saturation = initial_saturation
    best_diff = initial_diff
    best_comb = None

    print(f"Initial - Kernel size {initial_kernel_size} => Saturation: {initial_saturation:.4f} (diff: {initial_diff:.4f})")

    iteration = 0
    previous_diff = initial_diff

    while iteration < max_iterations:
        iteration += 1

        # Determine step based on current best saturation gap
        gap = target_sat - best_saturation
        direction = 1 if gap > 0 else -1
        step = _determine_step_size(gap)
        test_kernel = best_kernel_size + direction * step

        if test_kernel <= 0:
            break

        saturation_test, comb_test = compute_saturation_parallel(domain, test_kernel, theta, num_workers)
        diff_test = abs(saturation_test - target_sat)
        print(f"Iteration {iteration} - Kernel size {test_kernel} => Saturation: {saturation_test:.4f} (diff: {diff_test:.4f})")

        if diff_test > previous_diff:
            print(f"Difference increased from {previous_diff:.4f} to {diff_test:.4f}. Stopping and using previous best result.")
            break

        improvement = previous_diff - diff_test
        # If improvement is negligible (<0.01) *and* we are already within 0.02 of the
        # target, accept the current result and stop. Otherwise keep iterating to try
        # and reduce the remaining error.
        if improvement < 0.01:
            if diff_test <= 0.02:
                print(
                    f"Improvement ({improvement:.4f}) < 0.01 and diff ({diff_test:.4f}) ≤ 0.02. "
                    "Saving current result as final."
                )
                best_kernel_size = test_kernel
                best_saturation = saturation_test
                best_diff = diff_test
                best_comb = comb_test.copy()
                break
            else:
                print(
                    f"Improvement ({improvement:.4f}) < 0.01 but diff ({diff_test:.4f}) > 0.02. "
                    "Continuing search."
                )

        if diff_test < best_diff:
            best_kernel_size = test_kernel
            best_saturation = saturation_test
            best_diff = diff_test
            best_comb = comb_test.copy()
            print(f"Found better kernel size: {test_kernel}")

        previous_diff = diff_test

    if best_comb is None:
        _, best_comb = compute_saturation_parallel(domain, best_kernel_size, theta, num_workers)

    return best_kernel_size, best_saturation, best_comb


def read_config(config_file: str = "input.txt") -> dict:
    """Read configuration, searching both CWD and script directory when a relative path is given."""
    config: dict = {}

    # Build candidate paths
    cfg_path = Path(config_file)
    candidates: list[Path] = []
    if cfg_path.is_absolute():
        candidates.append(cfg_path)
    else:
        candidates.append(Path.cwd() / cfg_path)
        candidates.append(Path(__file__).resolve().parent / cfg_path)

    for candidate in candidates:
        try:
            with open(candidate, "r") as file:
                config_base_dir = Path(candidate).parent
                for line in file:
                    # Strip whitespace and inline comments
                    line = line.strip()
                    if not line:
                        continue
                    if "#" in line:
                        line = line.split("#", 1)[0].strip()
                    if not line:
                        continue
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key, value = key.strip(), value.strip()
                        # Guard against inline comments on the value side as well
                        if "#" in value:
                            value = value.split("#", 1)[0].strip()
                        if key == "filename":
                            # Resolve path relative to the config file location if not absolute
                            file_path = Path(value)
                            if not file_path.is_absolute():
                                file_path = (config_base_dir / file_path).resolve()
                            config[key] = str(file_path)
                        elif key in ["filesize_x", "filesize_y", "filesize_z"]:
                            config[key] = int(value)
                        elif key == "theta":
                            config[key] = float(value)
                        elif key == "target_saturations":
                            config[key] = [float(x.strip()) for x in value.split(",")]
                        elif key == "tol":
                            config[key] = float(value)
                        elif key == "num_threads":
                            config[key] = int(value)
                        else:
                            print(f"Warning: Unknown parameter '{key}' in config file")
            print(f"input parameters loaded from {candidate}")
            return config
        except FileNotFoundError:
            continue

    # If none of the candidates worked
    search_info = " or ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find '{config_file}'. Searched: {search_info}")


def print_config(config: dict):
    print("input Parameters:")
    print(f"Input file: {config['filename']}")
    print(f"File size: {config['filesize_x']} x {config['filesize_y']} x {config['filesize_z']}")
    print(f"Theta: {config['theta']} degrees")
    print(f"Target saturations: {config['target_saturations']}")
    print(f"Tolerance: {config['tol']}")
    print(f"Num threads: {config['num_threads']}")
    print("=" * 50)


# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------

def main():
    start_time = time.time()

    config = read_config()
    filename = config["filename"]
    filesize = [config["filesize_z"], config["filesize_y"], config["filesize_x"]]
    total_elements = filesize[0] * filesize[1] * filesize[2]

    num_workers = config.get("num_threads", mp.cpu_count())

    print_config(config)

    print(f"Loading {filename}...")
    with open(filename, "rb") as file:
        raw_data = np.fromfile(file, dtype=np.uint8, count=total_elements)

    domain = raw_data.reshape((filesize[0], filesize[1], filesize[2]))
    del raw_data

    print("Image loaded successfully")
    print(f"Porous domain porosity: {np.sum(domain == 0) / ((np.sum(domain == 1))+(np.sum(domain==0)))}")
    print("=" * 50)

    theta = config["theta"]
    target_saturations = config["target_saturations"]

    for target_sat in target_saturations:
        print(f"Processing target saturation: {target_sat}")
        print("=" * 25)
        print("Selecting best kernel size:")

        num_slices = filesize[0]
        num_to_take = 10 if num_slices >= 10 else 5
        mid_index = num_slices // 2
        start_index = max(0, mid_index - num_to_take // 2)
        end_index = min(num_slices, start_index + num_to_take)
        domain_test = domain[start_index:end_index]

        best_kernel_size, best_diff = find_best_kernel_size(domain_test, target_sat, theta, num_workers)
        print(f"Selected kernel size: {best_kernel_size} (best diff: {best_diff:.4f})")
        print("=" * 50)

        print("\nApplying best kernel size to full domain:")
        saturation_full, comb_full = compute_saturation_parallel(domain, best_kernel_size, theta, num_workers)
        print(f"Full domain results with kernel size {best_kernel_size}:")
        print(f"Full domain saturation: {saturation_full:.4f}")
        print(f"Difference from target saturation: {abs(saturation_full - target_sat):.4f}")

        if abs(saturation_full - target_sat) > 0.02:
            print("\nContinuing iteration to find better saturation...")
            best_kernel_size, saturation_full, comb_full = refine_kernel_size(
                domain, best_kernel_size, saturation_full, abs(saturation_full - target_sat), target_sat, theta, num_workers
            )

        output_filename = f"result_sat{saturation_full:.4f}_theta{theta}.raw"
        comb_full.astype(np.uint8).tofile(output_filename)
        print(f"\nSaved result to: {output_filename}")
        print(f"Final saturation: {saturation_full:.4f}")
        print(f"Final difference from target: {abs(saturation_full - target_sat):.4f}")
        print("=" * 50 + "\n")

        del comb_full, domain_test
        gc.collect()

    # Runtime reporting
    elapsed = time.time() - start_time
    print("=" * 50)
    if elapsed < 60:
        print(f"Total runtime: {elapsed:.2f} seconds")
    elif elapsed < 3600:
        m, s = divmod(elapsed, 60)
        print(f"Total runtime: {int(m)} minutes {s:.2f} seconds")
    else:
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)
        print(f"Total runtime: {int(h)} hours {int(m)} minutes {s:.2f} seconds")
    print("=" * 50)


# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Cross-platform safe start method
    if platform.system() == "Windows":
        mp.set_start_method("spawn", force=True)
        mp.freeze_support()
    else:
        # Prefer fork on Unix-like for performance/memory
        mp.set_start_method("fork", force=True)
    main()