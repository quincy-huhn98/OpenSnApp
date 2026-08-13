# run_rom_reed.py
from rom_driver import run_pipeline_1g
from reed_problem import ReedProblem
from job_manager import JobManager
from pathlib import Path
import argparse
import os
import sys

python_root = os.environ.get("OPENSN_PYTHON_PATH")
if python_root:
    sys.path.insert(0, python_root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exe",
        type=str,
        default=None,
        help="OpenSn application executable (e.g. opensn, ./opensn, path/to/app)",
    )
    ap.add_argument("--nprocs", type=int, default=2)
    ap.add_argument("--ntrain", type=int, default=100)
    ap.add_argument("--ntest", type=int, default=10)
    ap.add_argument("--mipod", action="store_true", help="Run optional MIPOD testing.")
    args = ap.parse_args()

    example_root = Path(__file__).resolve().parent
    os.chdir(example_root)

    # Pass executable into the JobManager
    jm = JobManager(opensn_exe=args.exe)

    problem = ReedProblem(
        example_root,
        nprocs=args.nprocs,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )

    run_pipeline_1g(problem, example_root, jm, run_mipod=args.mipod)
    problem.plot_results(include_mipod=args.mipod)


if __name__ == "__main__":
    main()
