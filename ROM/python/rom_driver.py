from pathlib import Path
import numpy as np
from scipy.stats import qmc

from AS import ActiveSubspace


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
    args = ["-p", f"phase={repr(phase)}", "-p", f"pid={pid}"]

    if pvec is not None:
        for i, v in enumerate(np.asarray(pvec, dtype=float).ravel()):
            args.extend(["-p", f"p{i}={float(v)}"])

    args.extend(["-p", f"saveh5={save_h5}"])
    return args


def _run_one(problem, jm, workdir, phase, pid, pvec=None, save_h5=False, deck=None):
    opensn_args = make_opensn_args(phase, pid, pvec, save_h5)

    jm.run(
        input_file=str(deck if deck is not None else problem.deck_path),
        nprocs=problem.nprocs,
        workdir=str(workdir),
        opensn_args=opensn_args,
        stream_output=True,
    )


def _run_many(problem, jm, workdir, phase, dataset, start_pid=0, save_h5=False):
    for i, pvec in enumerate(dataset):
        pid = start_pid + i
        problem.update_xs(pvec)
        _run_one(problem, jm, workdir, phase, pid, pvec, save_h5)


def _run_many_with_interpolation_points(
    problem,
    jm,
    workdir,
    phase,
    physical_dataset,
    interpolation_dataset,
    start_pid=0,
    save_h5=False,
):
    if len(physical_dataset) != len(interpolation_dataset):
        raise ValueError("physical_dataset and interpolation_dataset must have the same length.")

    for i, (physical_pvec, interpolation_pvec) in enumerate(
        zip(physical_dataset, interpolation_dataset)
    ):
        pid = start_pid + i
        problem.update_xs(physical_pvec)
        _run_one(
            problem,
            jm,
            workdir,
            phase,
            pid,
            pvec=interpolation_pvec,
            save_h5=save_h5,
        )


def _problem_bounds(problem):
    """
    Prefer the problem-level bounds, because P58Problem intentionally overrides
    the automatically generated xs.py bounds with a curated domain.
    """
    if hasattr(problem, "bounds"):
        return np.asarray(problem.bounds, dtype=float)
    return np.asarray(problem.xs.bounds, dtype=float)


def _sample_physical_lhs(bounds, n_samples):
    bounds = np.asarray(bounds, dtype=float)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    sampler = qmc.LatinHypercube(d=bounds.shape[0])
    u = sampler.random(n_samples)
    return qmc.scale(u, bounds[:, 0], bounds[:, 1])


def _run_gradients(problem, jm, workdir, dataset):
    grad_deck = problem.workdir / "gradients_P58.py"

    for pid, pvec in enumerate(dataset):
        problem.update_xs(pvec)
        _run_one(problem, jm, workdir, "offline", pid, pvec, deck=grad_deck)


def _load_gradients(problem, n_samples):
    grads = []
    for i in range(n_samples):
        fname = problem.workdir / "data" / f"gradients_{i}.txt"
        grads.append(np.asarray(np.loadtxt(fname), dtype=float).ravel())
    return np.vstack(grads)


def _save_active_parameter_files(problem, physical_training, active_training):
    data_dir = problem.workdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(data_dir / "params.txt", physical_training)
    np.savetxt(data_dir / "params_AS.txt", active_training)


def run_active_subspace_pipeline(
    problem,
    repo_root,
    jm,
    n_gradients,
    active_rank=1,
):
    """
    Active-subspace ROM pipeline.

    Gradient samples are used only to estimate the active basis. They are not
    included in the ROM training library. The ROM training set and testing set
    are sampled separately using only active coordinates, i.e.

        x = W_active y,

    with no inactive-coordinate sampling.

    The FOM/offline calculations are always run with the reconstructed physical
    XS parameters. ROM interpolation uses active variables stored in
    data/params_AS.txt and passed during online evaluation as p0, p1, ...,
    p{active_rank-1}.
    """

    paths = ensure_problem_dirs(Path(repo_root))
    bounds = _problem_bounds(problem)

    ntrain = int(problem.ntrain)
    ntest = int(problem.ntest)
    n_grad = int(n_gradients)

    if n_grad < 1:
        raise ValueError("At least one gradient sample is required.")
    if active_rank < 1 or active_rank > bounds.shape[0]:
        raise ValueError("active_rank must be between 1 and the number of parameters.")

    # -----------------------------------
    # Step 1: sample gradient locations only
    # -----------------------------------
    grad_points = _sample_physical_lhs(bounds, n_grad)
    np.savetxt(problem.workdir / "data" / "gradient_params.txt", grad_points)

    # -----------------------------------
    # Step 2: run gradients
    # -----------------------------------
    print(f"[AS] Running gradients for {n_grad} samples")
    _run_gradients(problem, jm, paths["root"], grad_points)
    gradients = _load_gradients(problem, n_grad)

    # -----------------------------------
    # Step 3: compute active subspace
    # -----------------------------------
    print("[AS] Computing active subspace")
    active_subspace = ActiveSubspace(bounds)
    active_subspace.add_gradients(gradients)
    active_subspace.compute_subspace()
    active_subspace.set_rank(active_rank)
    problem.active_subspace = active_subspace

    # -----------------------------------
    # Step 4: sample ROM training points in the active subspace only
    # -----------------------------------
    print(f"[AS] Sampling {ntrain} active-subspace training points")
    physical_training, active_training, _ = active_subspace.make_active_training_set(
        ntrain,
        method="lhs",
        inactive_scale=0.0,
        reject_outside=True,
    )

    problem.training_set = physical_training
    _save_active_parameter_files(problem, physical_training, active_training)

    # -----------------------------------
    # Step 5: OFFLINE for AS-sampled training points only
    # -----------------------------------
    print("[AS] Running offline for active-subspace training points")
    _run_many(problem, jm, paths["root"], "offline", problem.training_set)

    # -----------------------------------
    # Step 6: MERGE uses exactly the AS-sampled training snapshots
    # -----------------------------------
    _run_one(
        problem,
        jm,
        paths["root"],
        "merge",
        pid=ntrain,
        pvec=np.ones(bounds.shape[0]),
    )

    # -----------------------------------
    # Step 7: SYSTEMS for AS-sampled physical training points
    # -----------------------------------
    _run_many(problem, jm, paths["root"], "systems", problem.training_set)

    # -----------------------------------
    # Step 8: sample testing points in the active subspace only
    # -----------------------------------
    print(f"[AS] Sampling {ntest} active-subspace testing points")
    physical_testing, active_testing, _ = active_subspace.make_active_training_set(
        ntest,
        method="lhs",
        inactive_scale=0.0,
        reject_outside=True,
    )

    problem.testing_set = physical_testing
    np.savetxt(problem.workdir / "data" / "test_params.txt", physical_testing)
    np.savetxt(problem.workdir / "data" / "test_params_AS.txt", active_testing)

    _run_many(problem, jm, paths["root"], "offline", problem.testing_set, save_h5=True)
    _run_many(problem, jm, paths["root"], "mipod", problem.testing_set, save_h5=True)

    # Online ROM interpolation uses active coordinates, while the XS file is
    # updated with the corresponding reconstructed physical testing point.
    _run_many_with_interpolation_points(
        problem,
        jm,
        paths["root"],
        "online",
        physical_dataset=problem.testing_set,
        interpolation_dataset=active_testing,
        save_h5=True,
    )


def run_pipeline(problem, repo_root, jm):
    paths = ensure_problem_dirs(Path(repo_root))

    problem.sample_training()

    _run_many(problem, jm, paths["root"], "offline", problem.training_set)

    _run_one(problem, jm, paths["root"], "merge",
             pid=problem.ntrain,
             pvec=np.ones_like(problem.training_set[0]))

    _run_many(problem, jm, paths["root"], "systems", problem.training_set)

    problem.sample_testing()

    _run_many(problem, jm, paths["root"], "offline",
              problem.testing_set, save_h5=True)

    _run_many(problem, jm, paths["root"], "mipod",
              problem.testing_set, save_h5=True)

    _run_many(problem, jm, paths["root"], "online",
              problem.testing_set, save_h5=True)
