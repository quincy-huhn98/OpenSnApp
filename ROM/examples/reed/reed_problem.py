from pathlib import Path
import utils
import plotting
import numpy as np


class ReedProblem:
    def __init__(self, workdir, nprocs=2, ntrain=100, ntest=10):
        self.workdir = Path(workdir)
        self.deck_path = self.workdir / "base_reed.py"

        self.nprocs = nprocs
        self.ntrain = ntrain
        self.ntest = ntest

    def sample_training(self):
        bounds = [[0.0,1.0],[0.0,1.0]]

        self.training_set = utils.sample_parameter_space(bounds, self.ntrain)

        params_path = self.workdir / "data" / "params.txt"
        np.savetxt(str(params_path), self.training_set)


    def sample_testing(self):
        self.testing_set = np.random.uniform(0,1,[self.ntest,2])

        params_path = self.workdir / "data" / "test_params.txt"
        np.savetxt(str(params_path), self.testing_set)

    def update_xs(self):
        print("Reed problem uses SimpleOneGroupXS, use run_pipeline_1g")


    def plot_results(self):
        plotting.plot_sv(num_groups=1)
        errors = []
        speedups = []
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
            speedups.append(fom_time/rom_time)

        print("Avg Error ", np.mean(errors))
        np.savetxt(str(results_dir / "errors.txt"), errors)
        print("Avg Speedup ", np.mean(speedups))
        np.savetxt(str(results_dir / "speedups.txt"), speedups)

