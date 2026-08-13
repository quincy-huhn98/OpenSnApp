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
    if getattr(problem, "test_mode", False):
        opensn_args.extend(["-p", "test_mode=True"])
    if getattr(problem, "use_nlke", False):
        opensn_args.extend(["-p", "use_nlke=True"])

    jm.run(
        input_file=str(deck if deck is not None else problem.deck_path),
        nprocs=problem.nprocs,
        workdir=str(workdir),
        opensn_args=opensn_args,
        stream_output=True,
        check=True,
    )


def _run_many(problem, jm, workdir, phase, dataset, start_pid=0, save_h5=False):
    for pid, pvec in enumerate(dataset):
        if pid >= start_pid:
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
    if hasattr(problem, "bounds"):
        return np.asarray(problem.bounds, dtype=float)
    return np.asarray(problem.xs.bounds, dtype=float)


def _sample_physical_lhs(bounds, n_samples, seed=None):
    bounds = np.asarray(bounds, dtype=float)
    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    sampler = qmc.LatinHypercube(d=bounds.shape[0], seed=seed)
    u = sampler.random(n_samples)
    return qmc.scale(u, bounds[:, 0], bounds[:, 1])


def _gradient_deck(problem):
    if hasattr(problem, "gradient_deck_path"):
        return Path(problem.gradient_deck_path)
    return problem.workdir / "gradients_P58.py"


def _run_gradients(problem, jm, workdir, dataset):
    grad_deck = _gradient_deck(problem)

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


def _load_active_subspace(problem, bounds, active_rank):
    active_subspace = ActiveSubspace(bounds)

    active_subspace.evals = np.atleast_1d(
        np.loadtxt(problem.workdir / "results" / "AS_values.txt")
    )
    active_subspace.evecs = np.asarray(
        np.loadtxt(problem.workdir / "results" / "AS_vectors.txt"),
        dtype=float,
    )

    if active_subspace.evecs.ndim == 1:
        active_subspace.evecs = active_subspace.evecs[:, None]

    active_subspace.set_rank(active_rank)
    return active_subspace


def run_active_subspace_pipeline(
    problem,
    repo_root,
    jm,
    n_gradients,
    active_rank=1,
    systems_restart=False,
    run_mipod=False,
    random_seed=None,
):
    paths = ensure_problem_dirs(Path(repo_root))
    if hasattr(problem, "prepare_inputs"):
        problem.prepare_inputs()
    bounds = _problem_bounds(problem)

    ntrain = int(problem.ntrain)
    ntest = int(problem.ntest)
    n_grad = int(n_gradients)
    seed_sequence = np.random.SeedSequence(random_seed)
    gradient_seed, training_seed, testing_seed = (
        int(seed.generate_state(1)[0]) for seed in seed_sequence.spawn(3)
    )
    if systems_restart:
        problem.load_training()
        _run_one(
            problem,
            jm,
            paths["root"],
            "merge",
            pid=ntrain,
            pvec=np.ones(bounds.shape[0]),
        )

        active_subspace = _load_active_subspace(problem, bounds, active_rank)
        problem.active_subspace = active_subspace

        _run_many(
            problem,
            jm,
            paths["root"],
            "systems",
            problem.training_set,
            start_pid=0,
        )
    else:
        if n_grad < 1:
            raise ValueError("At least one gradient sample is required.")
        if active_rank < 1 or active_rank > bounds.shape[0]:
            raise ValueError("active_rank must be between 1 and the number of parameters.")

        grad_points = _sample_physical_lhs(bounds, n_grad, seed=gradient_seed)
        np.savetxt(problem.workdir / "data" / "gradient_params.txt", grad_points)

        print(f"[AS] Running gradients for {n_grad} samples")
        _run_gradients(problem, jm, paths["root"], grad_points)
        gradients = _load_gradients(problem, n_grad)

        print("[AS] Computing active subspace")
        active_subspace = ActiveSubspace(bounds)
        active_subspace.add_gradients(gradients)
        active_subspace.compute_subspace()
        active_subspace.set_rank(active_rank)
        problem.active_subspace = active_subspace

        print(f"[AS] Sampling {ntrain} active-subspace training points")
        physical_training, active_training, _ = active_subspace.make_active_training_set(
            ntrain,
            method="lhs",
            reject_outside=True,
            seed=training_seed,
        )

        problem.training_set = physical_training
        _save_active_parameter_files(problem, physical_training, active_training)

        print("[AS] Running offline for active-subspace training points")
        _run_many(problem, jm, paths["root"], "offline", problem.training_set)

        _run_one(
            problem,
            jm,
            paths["root"],
            "merge",
            pid=ntrain,
            pvec=np.ones(bounds.shape[0]),
        )

        _run_many(problem, jm, paths["root"], "systems", problem.training_set)

    print(f"[AS] Sampling {ntest} active-subspace testing points")
    physical_testing, active_testing, _ = active_subspace.make_active_training_set(
        ntest,
        method="lhs",
        reject_outside=True,
        seed=testing_seed,
    )

    problem.testing_set = physical_testing
    np.savetxt(problem.workdir / "data" / "test_params.txt", physical_testing)
    np.savetxt(problem.workdir / "data" / "test_params_AS.txt", active_testing)

    _run_many(problem, jm, paths["root"], "offline", problem.testing_set, save_h5=True)
    if run_mipod:
        _run_many(problem, jm, paths["root"], "mipod", problem.testing_set, save_h5=True)

    _run_many_with_interpolation_points(
        problem,
        jm,
        paths["root"],
        "online",
        physical_dataset=problem.testing_set,
        interpolation_dataset=active_testing,
        save_h5=True,
    )


def run_pipeline(problem, repo_root, jm, systems_restart=False, run_mipod=False):
    paths = ensure_problem_dirs(Path(repo_root))
    if hasattr(problem, "prepare_inputs"):
        problem.prepare_inputs()

    if systems_restart:
        problem.load_training()

        _run_one(
            problem,
            jm,
            paths["root"],
            "merge",
            pid=problem.ntrain,
            pvec=np.ones_like(problem.training_set[0]),
        )

        # , start_pid=systems_restart)
        _run_many(problem, jm, paths["root"], "systems", problem.training_set)
    else:
        problem.sample_training()

        _run_many(problem, jm, paths["root"], "offline", problem.training_set)

        _run_one(
            problem,
            jm,
            paths["root"],
            "merge",
            pid=problem.ntrain,
            pvec=np.ones_like(problem.training_set[0]),
        )

        _run_many(problem, jm, paths["root"], "systems", problem.training_set)

    problem.sample_testing()

    _run_many(problem, jm, paths["root"], "offline", problem.testing_set, save_h5=True)
    if run_mipod:
        _run_many(problem, jm, paths["root"], "mipod", problem.testing_set, save_h5=True)
    _run_many(problem, jm, paths["root"], "online", problem.testing_set, save_h5=True)


def _run_many_1g(problem, jm, workdir, phase, dataset, save_h5=False):
    """Run each single-group problem with command-line cross-section updates."""
    for pid, pvec in enumerate(dataset):
        _run_one(problem, jm, workdir, phase=phase, pid=pid, pvec=pvec, save_h5=save_h5)


def run_pipeline_1g(problem, repo_root, jm, run_mipod=False):
    """Run each ROM phase while passing cross sections to the input file."""
    paths = ensure_problem_dirs(Path(repo_root))

    problem.sample_training()

    # OFFLINE training
    _run_many_1g(problem, jm, workdir=paths["root"], phase="offline", dataset=problem.training_set)

    # MERGE
    _run_one(
        problem,
        jm,
        workdir=paths["root"],
        phase="merge",
        pid=problem.ntrain,
        pvec=np.ones_like(
            problem.training_set[0]))

    # SYSTEMS
    _run_many_1g(problem, jm, workdir=paths["root"], phase="systems", dataset=problem.training_set)

    problem.sample_testing()

    # OFFLINE testing (save HDF5)
    _run_many_1g(
        problem,
        jm,
        workdir=paths["root"],
        phase="offline",
        dataset=problem.testing_set,
        save_h5=True)

    # Optional minimally invasive POD testing (save HDF5)
    if run_mipod:
        _run_many_1g(
            problem,
            jm,
            workdir=paths["root"],
            phase="mipod",
            dataset=problem.testing_set,
            save_h5=True)

    # ONLINE testing (save HDF5)
    _run_many_1g(
        problem,
        jm,
        workdir=paths["root"],
        phase="online",
        dataset=problem.testing_set,
        save_h5=True)
