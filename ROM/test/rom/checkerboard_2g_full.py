#!/usr/bin/env python3
"""Exercise a two-rank, two-dimensional, two-group source ROM and MI-POD."""

import numpy as np

from integration_test import isolated_example, relative_flux_error, rom_executable


with isolated_example("2gcheckerboard") as workdir:
    from checkerboard_problem_2g import CheckerboardProblem2G
    from job_manager import JobManager
    from rom_driver import run_pipeline
    from utils import load_2d_flux

    problem = CheckerboardProblem2G(
        workdir, nprocs=2, ntrain=10, ntest=1, random_seed=20260812
    )
    problem.bounds = [[0.74, 0.76], [9.9, 10.1]]
    problem.test_mode = True

    manager = JobManager(opensn_exe=rom_executable())
    run_pipeline(problem, workdir, manager, run_mipod=True)

    output_dir = workdir / "output"
    _, _, fom_flux, _ = load_2d_flux(
        str(output_dir / "fom_0_{}.h5"), range(problem.nprocs)
    )
    _, _, rom_flux, _ = load_2d_flux(
        str(output_dir / "rom_0_{}.h5"), range(problem.nprocs)
    )
    _, _, mipod_flux, _ = load_2d_flux(
        str(output_dir / "mipod_0_{}.h5"), range(problem.nprocs)
    )
    relative_error = relative_flux_error(fom_flux, rom_flux)
    mipod_completed = np.all(np.isfinite(np.asarray(mipod_flux))) and (
        workdir / "results" / "mipod_time_0.txt"
    ).exists()

print(f"ROM_CHECKERBOARD_2G_REL_ERROR={relative_error:.16e}")
print(f"ROM_CHECKERBOARD_2G_MIPOD_OK={float(mipod_completed)}")
