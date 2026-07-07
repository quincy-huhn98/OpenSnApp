from pathlib import Path
import argparse
import os
import sys

python_root = os.environ.get("OPENSN_PYTHON_PATH")
if python_root:
    sys.path.insert(0, python_root)

from job_manager import JobManager
from c5g7_problem import C5G7Problem
from rom_driver import run_active_subspace_pipeline, run_pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", type=str, default=None)
    ap.add_argument("--system", type=str, default="auto")

    ap.add_argument("--nprocs", type=int, default=48)
    ap.add_argument("--ntrain", type=int, default=100)
    ap.add_argument("--ntest", type=int, default=10)

    ap.add_argument("--active-subspace", action="store_true")
    ap.add_argument("--n-gradients", type=int, default=20)
    ap.add_argument("--active-rank", type=int, default=1)
    ap.add_argument("--plot-only", action="store_true")

    args = ap.parse_args()

    repo_root = Path.cwd()

    jm = JobManager(
        system=args.system,
        opensn_exe=args.exe,
    )

    problem = C5G7Problem(
        repo_root,
        nprocs=args.nprocs,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )

    if not args.plot_only:
        if args.active_subspace:
            run_active_subspace_pipeline(
                problem,
                repo_root,
                jm,
                n_gradients=args.n_gradients,
                active_rank=args.active_rank,
                systems_restart=1
            )
        else:
            run_pipeline(problem, repo_root, jm)

    problem.plot_results()


if __name__ == "__main__":
    main()
