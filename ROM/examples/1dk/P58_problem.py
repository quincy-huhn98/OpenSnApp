from pathlib import Path
import utils
import plotting
import numpy as np
import xs


class P58Problem:
    def __init__(self, workdir, nprocs=2, ntrain=100, ntest=10):
        self.workdir = Path(workdir)
        self.deck_path = self.workdir / "base_P58.py"

        self.nprocs = nprocs
        self.ntrain = ntrain
        self.ntest = ntest

        self.xs = xs.CrossSections(
            [
                {
                    "in_file": "URRb_base.txt",
                    "out_file": "data/URRb.xs",
                    "kind": "fission",
                },
            ],
            frac=0.2,
            transfer_tol=1.0e-14,
            param_mode="entrywise",
        )

        # Preserve the original curated P58 parameter domain rather than
        # adopting auto-generated relative bounds from xs.py.
        self.bounds = [
            [0.000836,0.001648],  # sigma_f[0]
            [0.029564,0.057296],  # sigma_f[1]
            [0.001104,0.001472],  # sigma_c[0]
            [0.024069,0.029244],  # sigma_c[1]
            [0.83807,0.83892],    # S[0,0]
            [0.04536,0.04635],    # S[0,1]
            [0.000767,0.00116],   # S[1,0]
            [2.8751,2.9183],      # S[1,1]
        ]
        if len(self.bounds) != self.xs.n_params:
            raise ValueError(
                "P58 bounds define {} parameters, but xs.py expects {}."
                .format(len(self.bounds), self.xs.n_params)
            )

    def sample_training(self):
        self.training_set = utils.sample_LHS(self.bounds, self.ntrain)

        params_path = self.workdir / "data" / "params.txt"
        np.savetxt(str(params_path), self.training_set)
        # Also move water xs file to data
        with open("H2O_mg_base.txt", "rb") as f_src, open("data/H2O_mg.xs", "wb") as f_dst:
            f_dst.write(f_src.read())

    def load_training(self):
        params_path = self.workdir / "data" / "params.txt"
        self.training_set = np.loadtxt(str(params_path))

    def sample_testing(self):
        self.testing_set = utils.sample_test(self.bounds, self.ntest)

        params_path = self.workdir / "data" / "test_params.txt"
        np.savetxt(str(params_path), self.testing_set)

    def update_xs(self, pvec):
        self.xs.write_sample(pvec)

    def plot_results(self):
        plotting.plot_sv(num_groups=2)
        errors = []
        k_errors = []
        speedups = []
        for i in range(self.ntest):
            results_dir = self.workdir / "results"
            rom_time = np.loadtxt(str(results_dir / "online_time_{}.txt".format(i)))
            fom_time = np.loadtxt(str(results_dir / "offline_time_{}.txt".format(i)))

            output_dir = self.workdir / "output"
            error = plotting.plot_1d_eigenvector(
                str(output_dir / ("fom_{}_".format(i) + "{}.h5")),
                str(output_dir / ("rom_{}_".format(i) + "{}.h5")),
                ranks=range(self.nprocs),
                pid=i)
            k_error = np.abs(
                np.loadtxt("output/fom_k_{}.txt".format(i))
                - np.loadtxt("output/rom_k_{}.txt".format(i))
            )

            k_errors.append(k_error)
            errors.append(error)
            speedups.append(fom_time / rom_time)

        print("Avg Eigenvector Error ", np.mean(errors))
        np.savetxt("results/errors.txt", errors)
        print("Avg k Error ", np.mean(k_errors) * 1e5, "pcm")
        np.savetxt("results/k_errors.txt", k_errors)
        print("Avg Speedup ", np.mean(speedups))
        np.savetxt("results/speedups.txt", speedups)
