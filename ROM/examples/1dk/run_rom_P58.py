from rom_driver import run_pipeline, run_active_subspace_pipeline
from P58_problem import P58Problem
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

    ap.add_argument("--exe", type=str, default=None)
    ap.add_argument("--nprocs", type=int, default=2)
    ap.add_argument("--ntrain", type=int, default=100)
    ap.add_argument("--ntest", type=int, default=10)

    # Active subspace options
    ap.add_argument("--active-subspace", action="store_true")
    ap.add_argument("--active-num-gradients", type=int, default=20)
    ap.add_argument("--active-rank", type=int, default=1)
    ap.add_argument("--mipod", action="store_true", help="Run optional MIPOD testing.")
    ap.add_argument(
        "--systems-restart",
        action="store_true",
        help="Restart at basis merge/system construction using existing training data.",
    )

    args = ap.parse_args()

    example_root = Path(__file__).resolve().parent
    os.chdir(example_root)

    jm = JobManager(opensn_exe=args.exe)

    problem = P58Problem(
        example_root,
        nprocs=args.nprocs,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )

    if args.active_subspace:
        run_active_subspace_pipeline(
            problem,
            example_root,
            jm,
            n_gradients=args.active_num_gradients,
            active_rank=args.active_rank,
            systems_restart=args.systems_restart,
            run_mipod=args.mipod,
        )
    else:
        run_pipeline(
            problem,
            example_root,
            jm,
            systems_restart=args.systems_restart,
            run_mipod=args.mipod,
        )

    problem.plot_results(include_mipod=args.mipod)


if __name__ == "__main__":
    main()
