#!/usr/bin/env python3
"""Exercise a seeded P58 NLKE pipeline and MI-POD."""

import numpy as np

from integration_test import isolated_example, relative_flux_error, rom_executable


with isolated_example("1dk") as workdir:
    from job_manager import JobManager
    from P58_problem import P58Problem
    from rom_driver import run_pipeline
    from utils import load_1d_flux

    problem = P58Problem(
        workdir, nprocs=1, ntrain=10, ntest=1, random_seed=20260812
    )
    nominal_xs = problem.xs.get_nominal_sample()
    problem.xs.bounds = [(0.98 * value, 1.02 * value) for value in nominal_xs]
    problem.test_mode = True
    problem.use_nlke = True

    manager = JobManager(opensn_exe=rom_executable())
    run_pipeline(problem, workdir, manager, run_mipod=True)

    output_dir = workdir / "output"
    fom_k = float(np.loadtxt(output_dir / "test_fom_k_0.txt"))
    ommi_k = float(np.loadtxt(output_dir / "rom_k_0.txt"))
    mipod_k = float(np.loadtxt(output_dir / "mipod_k_0.txt"))
    ommi_k_relative_error = abs(ommi_k - fom_k) / abs(fom_k)
    mipod_k_relative_error = abs(mipod_k - fom_k) / abs(fom_k)

    _, fom_flux, _ = load_1d_flux(
        str(output_dir / "fom_0_{}.h5"), range(problem.nprocs)
    )
    _, ommi_flux, _ = load_1d_flux(
        str(output_dir / "rom_0_{}.h5"), range(problem.nprocs)
    )
    _, mipod_flux, _ = load_1d_flux(
        str(output_dir / "mipod_0_{}.h5"), range(problem.nprocs)
    )
    ommi_flux_relative_error = relative_flux_error(
        fom_flux, ommi_flux, normalize=True
    )
    mipod_flux_relative_error = relative_flux_error(
        fom_flux, mipod_flux, normalize=True
    )

print(f"ROM_P58_NLKE_OMMI_K_REL_ERROR={ommi_k_relative_error:.16e}")
print(f"ROM_P58_NLKE_OMMI_FLUX_REL_ERROR={ommi_flux_relative_error:.16e}")
print(f"ROM_P58_NLKE_MIPOD_K_REL_ERROR={mipod_k_relative_error:.16e}")
print(f"ROM_P58_NLKE_MIPOD_FLUX_REL_ERROR={mipod_flux_relative_error:.16e}")
