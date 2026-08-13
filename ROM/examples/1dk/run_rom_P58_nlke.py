#!/usr/bin/env python3
"""Run the complete P58 active-subspace pipeline with the NLKE ROM solver."""

import argparse
import os
import sys
from pathlib import Path

from job_manager import JobManager
from P58_problem import P58Problem
from rom_driver import run_active_subspace_pipeline

python_root = os.environ.get("OPENSN_PYTHON_PATH")
if python_root:
    sys.path.insert(0, python_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=str, default=None)
    parser.add_argument("--nprocs", type=int, default=2)
    parser.add_argument("--ntrain", type=int, default=20)
    parser.add_argument("--ntest", type=int, default=2)
    parser.add_argument("--num-gradients", type=int, default=10)
    parser.add_argument("--active-rank", type=int, default=1)
    parser.add_argument("--mipod", action="store_true")
    parser.add_argument("--systems-restart", action="store_true")
    args = parser.parse_args()

    example_root = Path(__file__).resolve().parent
    os.chdir(example_root)

    problem = P58Problem(
        example_root,
        nprocs=args.nprocs,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )
    problem.use_nlke = True

    manager = JobManager(opensn_exe=args.exe)
    run_active_subspace_pipeline(
        problem,
        example_root,
        manager,
        n_gradients=args.num_gradients,
        active_rank=args.active_rank,
        systems_restart=args.systems_restart,
        run_mipod=args.mipod,
    )
    problem.plot_results(include_mipod=args.mipod)


if __name__ == "__main__":
    main()
