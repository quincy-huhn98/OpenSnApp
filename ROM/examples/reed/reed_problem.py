from pathlib import Path
import plotting
import numpy as np


class ReedProblem:
    def __init__(self, workdir, nprocs=2, ntrain=100, ntest=10, random_seed=None):
        self.workdir = Path(workdir)
        self.deck_path = self.workdir / "base_reed.py"

        self.nprocs = nprocs
        self.ntrain = ntrain
        self.ntest = ntest
        self.rng = np.random.default_rng(random_seed)

    def sample_training(self):
        bounds = [[0.0, 1.0], [0.0, 1.0]]

        self.training_set = self.rng.uniform(
            np.asarray(bounds)[:, 0], np.asarray(bounds)[:, 1], (self.ntrain, 2)
        )

        params_path = self.workdir / "data" / "params.txt"
        np.savetxt(str(params_path), self.training_set)

    def sample_testing(self):
        self.testing_set = self.rng.uniform(0, 1, [self.ntest, 2])

        params_path = self.workdir / "data" / "test_params.txt"
        np.savetxt(str(params_path), self.testing_set)

    def update_xs(self):
        print("Reed problem uses SimpleOneGroupXS, use run_pipeline_1g")

    def plot_results(self, include_mipod=False):
        plotting.plot_sv(num_groups=1)
        errors = []
        speedups = []
        mipod_errors = []
        mipod_speedups = []
        for i in range(self.ntest):
            results_dir = self.workdir / "results"
            rom_time = np.loadtxt(str(results_dir / "online_time_{}.txt".format(i)))
            fom_time = np.loadtxt(str(results_dir / "offline_time_{}.txt".format(i)))

            output_dir = self.workdir / "output"
            error = plotting.plot_1d_flux(
                str(output_dir / ("fom_{}_".format(i) + "{}.h5")),
                str(output_dir / ("rom_{}_".format(i) + "{}.h5")),
                ranks=range(self.nprocs),
                pid=i)

            errors.append(error)
            speedups.append(fom_time / rom_time)

            if include_mipod:
                mipod_time = np.loadtxt(
                    str(results_dir / "mipod_time_{}.txt".format(i))
                )
                mipod_error = plotting.plot_1d_flux(
                    str(output_dir / ("fom_{}_".format(i) + "{}.h5")),
                    str(output_dir / ("mipod_{}_".format(i) + "{}.h5")),
                    ranks=range(self.nprocs),
                    prefix="reed_mipod",
                    pid=i,
                )
                mipod_errors.append(mipod_error)
                mipod_speedups.append(fom_time / mipod_time)

        print("Avg Error ", np.mean(errors))
        np.savetxt(str(results_dir / "errors.txt"), errors)
        print("Avg Speedup ", np.mean(speedups))
        np.savetxt(str(results_dir / "speedups.txt"), speedups)
        if include_mipod:
            print("Avg MI-POD Error ", np.mean(mipod_errors))
            np.savetxt(str(results_dir / "mipod_errors.txt"), mipod_errors)
            print("Avg MI-POD Speedup ", np.mean(mipod_speedups))
            np.savetxt(str(results_dir / "mipod_speedups.txt"), mipod_speedups)
