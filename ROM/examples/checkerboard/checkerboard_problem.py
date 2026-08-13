from pathlib import Path
import utils
import plotting
import numpy as np


class CheckerboardProblem:
    def __init__(self, workdir, five_param=False, nprocs=4, ntrain=100, ntest=10):
        self.workdir = Path(workdir)
        self.deck_path = self.workdir / "base_checkerboard.py"

        self.nprocs = nprocs
        self.ntrain = ntrain
        self.ntest = ntest
        self.five_param = five_param

    def sample_training(self):
        if self.five_param:
            bounds = [[0, 5.0], [0.5, 1.5], [7.5, 12.5], [0.0, 0.5], [0.1, 1]]
        else:
            bounds = [[0, 5.0], [0.5, 1.5], [7.5, 12.5], [0.0, 0.5]]

        self.training_set = utils.sample_parameter_space(bounds, self.ntrain)

        params_path = self.workdir / "data" / "params.txt"
        np.savetxt(str(params_path), self.training_set)

    def sample_testing(self):
        test_scatt_1 = np.random.uniform(0, 5.0, self.ntest)
        test_scatt_2 = np.random.uniform(0.5, 1.5, self.ntest)
        test_abs_1 = np.random.uniform(7.5, 12.5, self.ntest)
        test_abs_2 = np.random.uniform(0.0, 0.5, self.ntest)

        test = np.append(test_scatt_1[:, np.newaxis], test_scatt_2[:, np.newaxis], axis=1)
        test = np.append(test, test_abs_1[:, np.newaxis], axis=1)
        self.testing_set = np.append(test, test_abs_2[:, np.newaxis], axis=1)

        if self.five_param:
            test_q = np.random.uniform(0.1, 1, self.ntest)
            self.testing_set = np.append(self.testing_set, test_q[:, np.newaxis], axis=1)

        params_path = self.workdir / "data" / "test_params.txt"
        np.savetxt(str(params_path), self.testing_set)

    def update_xs(self):
        print("Checkerbaord problem uses SimpleOneGroupXS, use run_pipeline_1g")

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
            plotting.plot_2d_flux(str(output_dir / ("fom_{}_".format(i) + "{}.h5")),
                                  ranks=range(self.nprocs), prefix="fom", pid=i)
            plotting.plot_2d_flux(str(output_dir / ("rom_{}_".format(i) + "{}.h5")),
                                  ranks=range(self.nprocs), prefix="rom", pid=i)

            error = plotting.plot_2d_lineout(output_dir, ranks=range(self.nprocs), pid=i)

            errors.append(error)
            speedups.append(fom_time / rom_time)

            if include_mipod:
                mipod_time = np.loadtxt(
                    str(results_dir / "mipod_time_{}.txt".format(i))
                )
                plotting.plot_2d_flux(
                    str(output_dir / ("mipod_{}_".format(i) + "{}.h5")),
                    ranks=range(self.nprocs),
                    prefix="mipod",
                    pid=i,
                )
                mipod_error = plotting.plot_2d_lineout(
                    output_dir,
                    ranks=range(self.nprocs),
                    pid=i,
                    rom_prefix="mipod",
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
