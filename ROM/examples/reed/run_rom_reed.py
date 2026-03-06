# run_rom_reed.py
from pathlib import Path
import argparse
import os, sys

python_root = os.environ.get("OPENSN_PYTHON_PATH")
if python_root:
    sys.path.insert(0, python_root)

from job_manager import JobManager
from reed_problem import ReedProblem
from rom_driver import run_pipeline_1g

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exe",
        type=str,
        default=None,
        help="OpenSn application executable (e.g. opensn, ./opensn, path/to/app)",
    )
    ap.add_argument(
        "--system",
        type=str,
        default="auto",
        help="Execution system: auto, slurm, local, etc.",
    )
    args = ap.parse_args()

    repo_root = Path.cwd()

    # Pass executable into the JobManager
    jm = JobManager(
        system=args.system,
        opensn_exe=args.exe,
    )

    problem = ReedProblem(repo_root)

    run_pipeline_1g(problem, repo_root, jm)
    problem.plot_results()


if __name__ == "__main__":
    main()
