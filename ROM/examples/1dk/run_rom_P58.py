from pathlib import Path
import argparse
import os, sys

python_root = os.environ.get("OPENSN_PYTHON_PATH")
if python_root:
    sys.path.insert(0, python_root)

from job_manager import JobManager
from P58_problem import P58Problem
from rom_driver import run_pipeline, run_active_subspace_pipeline


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--exe", type=str, default=None)
    ap.add_argument("--system", type=str, default="auto")

    # Active subspace options
    ap.add_argument("--active-subspace", action="store_true")
    ap.add_argument("--active-num-gradients", type=int, default=20)
    ap.add_argument("--active-rank", type=int, default=1)

    args = ap.parse_args()

    repo_root = Path.cwd()

    jm = JobManager(
        system=args.system,
        opensn_exe=args.exe,
    )

    problem = P58Problem(repo_root, ntrain=100)

    if args.active_subspace:
        run_active_subspace_pipeline(
            problem,
            repo_root,
            jm,
            n_gradients=args.active_num_gradients,
            active_rank=args.active_rank,
        )
    else:
        run_pipeline(problem, repo_root, jm)

    problem.plot_results()


if __name__ == "__main__":
    main()
