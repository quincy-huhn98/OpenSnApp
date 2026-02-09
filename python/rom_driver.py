from pathlib import Path
import numpy as np


def ensure_problem_dirs(problem_root):
    paths = {
        "root": problem_root,
        "data": problem_root / "data",
        "basis": problem_root / "basis",
        "output": problem_root / "output",
        "results": problem_root / "results",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def make_opensn_args(phase, pid, pvec, save_h5=False):
    args = ["-p", "phase={}".format(repr(phase)),"-p", "pid={}".format(pid)]

    if pvec is not None:
        for i, v in enumerate(pvec):
            args.extend(["-p", "p{}={}".format(i, float(v))])

    if save_h5:
        args.extend(["-p", "saveh5=True"])

    return args


def _run_one(problem, jm, workdir, phase, pid, pvec=None, save_h5=False):
    opensn_args = make_opensn_args(phase=phase, pid=pid, pvec=pvec, save_h5=save_h5)

    res = jm.run(
        input_file=str(problem.deck_path),
        nprocs=problem.nprocs,
        workdir=str(workdir),
        opensn_args=opensn_args,
        stream_output=True,
    )

def _run_many(problem, jm, workdir, phase, dataset, save_h5=False):
    for pid, pvec in enumerate(dataset):
        problem.update_xs(pvec)
        _run_one(problem, jm, workdir, phase=phase, pid=pid, pvec=pvec, save_h5=save_h5)

def _run_many_1g(problem, jm, workdir, phase, dataset, save_h5=False):
    for pid, pvec in enumerate(dataset):
        _run_one(problem, jm, workdir, phase=phase, pid=pid, pvec=pvec, save_h5=save_h5)


def run_pipeline(problem, repo_root, jm):
    paths = ensure_problem_dirs(Path(repo_root))

    problem.sample_training()

    # OFFLINE training
    _run_many(problem, jm, workdir=paths["root"], phase="offline", dataset=problem.training_set)

    # MERGE
    _run_one(problem, jm, workdir=paths["root"], phase="merge", pid=problem.ntrain - 1, pvec=np.ones_like(problem.training_set[0]))

    # SYSTEMS
    _run_many(problem, jm, workdir=paths["root"], phase="systems", dataset=problem.training_set)

    problem.sample_testing()

    # OFFLINE testing (save HDF5)
    _run_many(problem, jm, workdir=paths["root"], phase="offline", dataset=problem.testing_set, save_h5=True)

    # ONLINE testing (save HDF5)
    _run_many(problem, jm, workdir=paths["root"], phase="online",  dataset=problem.testing_set, save_h5=True)

def run_pipeline_1g(problem, repo_root, jm):
    paths = ensure_problem_dirs(Path(repo_root))

    problem.sample_training()

    # OFFLINE training
    _run_many_1g(problem, jm, workdir=paths["root"], phase="offline", dataset=problem.training_set)

    # MERGE
    _run_one(problem, jm, workdir=paths["root"], phase="merge", pid=problem.ntrain - 1, pvec=np.ones_like(problem.training_set[0]))

    # SYSTEMS
    _run_many_1g(problem, jm, workdir=paths["root"], phase="systems", dataset=problem.training_set)

    problem.sample_testing()

    # OFFLINE testing (save HDF5)
    _run_many_1g(problem, jm, workdir=paths["root"], phase="offline", dataset=problem.testing_set, save_h5=True)

    # ONLINE testing (save HDF5)
    _run_many_1g(problem, jm, workdir=paths["root"], phase="online",  dataset=problem.testing_set, save_h5=True)
