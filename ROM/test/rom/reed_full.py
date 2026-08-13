#!/usr/bin/env python3
"""Check seeded Reed ROM predictions against the full-order model."""

import numpy as np

from integration_test import isolated_example, relative_flux_error, rom_executable


with isolated_example("reed") as workdir:
    from job_manager import JobManager
    from reed_problem import ReedProblem
    from rom_driver import run_pipeline_1g

    from utils import load_1d_flux

    problem = ReedProblem(
        workdir, nprocs=2, ntrain=50, ntest=1, random_seed=20260812
    )
    problem.test_mode = True
    manager = JobManager(opensn_exe=rom_executable())
    run_pipeline_1g(problem, workdir, manager, run_mipod=True)

    _, fom_flux, _ = load_1d_flux(
        str(workdir / "output" / "fom_0_{}.h5"), range(problem.nprocs)
    )
    _, rom_flux, _ = load_1d_flux(
        str(workdir / "output" / "rom_0_{}.h5"), range(problem.nprocs)
    )
    _, mipod_flux, _ = load_1d_flux(
        str(workdir / "output" / "mipod_0_{}.h5"), range(problem.nprocs)
    )
    relative_error = relative_flux_error(fom_flux, rom_flux)
    mipod_completed = np.all(np.isfinite(np.asarray(mipod_flux))) and (
        workdir / "results" / "mipod_time_0.txt"
    ).exists()

print(f"ROM_REED_REL_ERROR={relative_error:.16e}")
print(f"ROM_REED_MIPOD_OK={float(mipod_completed)}")
