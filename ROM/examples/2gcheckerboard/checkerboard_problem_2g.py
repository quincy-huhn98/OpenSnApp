from pathlib import Path
import numpy as np

import plotting
import utils
from xs import CrossSections


class CheckerboardProblem2G:
    def __init__(self, workdir, nprocs=4, ntrain=100, ntest=10):
        self.workdir = Path(workdir)
        self.deck_path = self.workdir / "base_2gcheckerboard.py"

        self.nprocs = nprocs
        self.ntrain = ntrain
        self.ntest = ntest

        self.xs = CrossSections(
            [
                {
                    "in_file": self.workdir / "scatterer_base.txt",
                    "out_file": self.workdir / "data" / "scatterer.xs",
                    "kind": "total",
                },
                {
                    "in_file": self.workdir / "absorber_base.txt",
                    "out_file": self.workdir / "data" / "absorber.xs",
                    "kind": "total",
                },
            ],
            param_mode="entrywise",
        )

        self.nominal_sample = self.xs.get_nominal_sample()
        self.full_bounds = list(self.xs.get_bounds())

        # Keep the original 2-parameter interface:
        #   pvec[0] -> scatterer downscatter entry S(0,1)
        #   pvec[1] -> absorber absorption sigma_a applied to both groups
        self._scatterer_downscatter_idx = self._find_total_transfer_index(
            self.workdir / "scatterer_base.txt", 0, 1
        )
        self._absorber_abs_indices = [
            self._find_total_abs_index(self.workdir / "absorber_base.txt", 0),
            self._find_total_abs_index(self.workdir / "absorber_base.txt", 1),
        ]

        # Override the active bounds to preserve the original checkerboard
        # ranges while still using CrossSections for the flattened XS layout.
        self.full_bounds[self._scatterer_downscatter_idx] = (0.5, 1.0)
        for idx in self._absorber_abs_indices:
            self.full_bounds[idx] = (7.5, 12.5)

        self.bounds = [
            list(self.full_bounds[self._scatterer_downscatter_idx]),
            list(self.full_bounds[self._absorber_abs_indices[0]]),
        ]

    def _find_total_abs_index(self, in_file, group):
        block = self.xs.get_block(in_file)
        return block["slice"].start + group

    def _find_total_transfer_index(self, in_file, gfrom, gto):
        block = self.xs.get_block(in_file)
        try:
            entry_idx = block["transfer_entries"].index((gfrom, gto))
        except ValueError as exc:
            raise ValueError(
                "Transfer entry ({}, {}) not found in {}.".format(gfrom, gto, in_file)
            ) from exc

        return block["slice"].start + block["G"] + entry_idx

    def _build_full_sample(self, pvec):
        full_sample = self.nominal_sample.copy()
        full_sample[self._scatterer_downscatter_idx] = pvec[0]
        for idx in self._absorber_abs_indices:
            full_sample[idx] = pvec[1]
        return full_sample

    def sample_training(self):
        self.training_set = utils.sample_parameter_space(self.bounds, self.ntrain)

        params_path = self.workdir / "data" / "params.txt"
        np.savetxt(str(params_path), self.training_set)

    def sample_testing(self):
        self.testing_set = utils.sample_test(self.bounds, self.ntest)

        params_path = self.workdir / "data" / "test_params.txt"
        np.savetxt(str(params_path), self.testing_set)

    def update_xs(self, pvec):
        self.xs.write_sample(self._build_full_sample(pvec))

    def plot_results(self):
        plotting.plot_sv(num_groups=2)
        errors = []
        speedups = []
        for i in range(self.ntest):
            results_dir = self.workdir / "results"
            rom_time = np.loadtxt(str(results_dir / "online_time_{}.txt".format(i)))
            fom_time = np.loadtxt(str(results_dir / "offline_time_{}.txt".format(i)))

            output_dir = self.workdir / "output"
            plotting.plot_2d_flux(
                str(output_dir / ("fom_{}_".format(i) + "{}.h5")),
                ranks=range(4),
                prefix="fom",
                pid=i,
            )
            plotting.plot_2d_flux(
                str(output_dir / ("rom_{}_".format(i) + "{}.h5")),
                ranks=range(4),
                prefix="rom",
                pid=i,
            )

            error = plotting.plot_2d_lineout(output_dir, ranks=range(4), pid=i)

            errors.append(error)
            speedups.append(fom_time / rom_time)

        print("Avg Error ", np.mean(errors))
        np.savetxt(str(results_dir / "errors.txt"), errors)
        print("Avg Speedup ", np.mean(speedups))
        np.savetxt(str(results_dir / "speedups.txt"), speedups)
