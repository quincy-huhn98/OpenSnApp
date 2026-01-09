from pathlib import Path
import numpy as np


class CrossSections:
    """
    Build a single parameter space across multiple XS files.

    Modes
    -----
    param_mode="entrywise"
        Original behavior:
          - total XS:   [sigma_a_vec, transfer_vals]
          - fission XS: [sigma_f_vec, sigma_c_vec, transfer_vals]

    param_mode="block_scales"
        Sample multiplicative scale factors instead of every individual entry:
          - total XS:   [sigma_a_scale, scatter_scale]
          - fission XS: [sigma_f_scale, sigma_c_scale, scatter_scale]

        Each scale multiplies the entire corresponding block.
        For example, sigma_f_scale multiplies all group values in sigma_f.
    """

    def __init__(self, xs_specs, frac=0.2, transfer_tol=0.0, param_mode="entrywise"):
        self.frac = frac
        self.transfer_tol = transfer_tol
        self.param_mode = param_mode

        if self.param_mode not in ("entrywise", "block_scales"):
            raise ValueError(
                "Unknown param_mode '{}'. Expected 'entrywise' or 'block_scales'.".format(
                    self.param_mode
                )
            )

        self.blocks = []
        self.bounds = []
        self.n_params = 0

        start = 0
        for spec in xs_specs:
            in_file = Path(spec["in_file"])
            out_file = Path(spec["out_file"])
            kind = spec["kind"]

            if kind == "total":
                data = self._read_total_xs(in_file)
                if self.param_mode == "entrywise":
                    x_nominal = self._flatten_total_data(data)
                else:
                    x_nominal = self._flatten_total_scales(data)

            elif kind == "fission":
                data = self._read_fission_xs(in_file)
                if self.param_mode == "entrywise":
                    x_nominal = self._flatten_fission_data(data)
                else:
                    x_nominal = self._flatten_fission_scales(data)

            else:
                raise ValueError(
                    "Unknown XS kind '{}'. Expected 'total' or 'fission'.".format(kind)
                )

            stop = start + len(x_nominal)

            block = {
                "in_file": in_file,
                "out_file": out_file,
                "kind": kind,
                "G": data["G"],
                "transfer_entries": data["transfer_entries"],
                "x_nominal": x_nominal,
                "bounds": self._relative_bounds(x_nominal),
                "slice": slice(start, stop),
                "n_params": len(x_nominal),
                "data": data,
            }

            self.blocks.append(block)
            self.bounds.extend(block["bounds"])
            start = stop

        self.n_params = start

    def get_bounds(self):
        return self.bounds

    def get_nominal_sample(self):
        return np.concatenate([block["x_nominal"] for block in self.blocks])

    def random_sample(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        vals = [rng.uniform(lo, hi) for (lo, hi) in self.bounds]
        return np.asarray(vals, dtype=float)

    def get_block(self, in_file):
        in_file = Path(in_file)
        for block in self.blocks:
            if block["in_file"] == in_file:
                return block
        raise KeyError("{} not found in CrossSections.".format(in_file))

    def get_block_sample(self, sample, in_file):
        block = self.get_block(in_file)
        return np.asarray(sample[block["slice"]], dtype=float)

    def write_file(self, sample, in_file):
        block = self.get_block(in_file)
        x_block = self.get_block_sample(sample, in_file)

        if block["kind"] == "total":
            if self.param_mode == "entrywise":
                sigma_a_vec, S = self._unflatten_total_data(x_block, block)
            else:
                sigma_a_vec, S = self._unflatten_total_scales(x_block, block)

            self._update_xs(
                block["in_file"],
                block["out_file"],
                sigma_a_vec,
                S,
            )

        elif block["kind"] == "fission":
            if self.param_mode == "entrywise":
                sigma_f_vec, sigma_c_vec, S = self._unflatten_fission_data(x_block, block)
            else:
                sigma_f_vec, sigma_c_vec, S = self._unflatten_fission_scales(x_block, block)

            self._update_fission_xs(
                block["in_file"],
                block["out_file"],
                sigma_f_vec,
                sigma_c_vec,
                S,
            )
        else:
            raise ValueError("Unknown XS kind '{}'.".format(block["kind"]))

    def write_sample(self, sample):
        sample = np.asarray(sample, dtype=float)
        if len(sample) != self.n_params:
            raise ValueError(
                "Sample has length {}, expected {}.".format(len(sample), self.n_params)
            )

        for block in self.blocks:
            self.write_file(sample, block["in_file"])

    # -------------------------------------------------------------------------
    # XS writers
    # -------------------------------------------------------------------------

    def _update_xs(self, in_file, out_file, sigma_a_vec, S):
        with open(in_file, "r") as f:
            lines = f.readlines()

        sigma_t_vec = np.zeros_like(sigma_a_vec, dtype=float)

        G = len(sigma_a_vec)
        for gfrom in range(G):
            sigma_t_vec[gfrom] = float(sigma_a_vec[gfrom])
            for gto in range(G):
                sigma_t_vec[gfrom] += float(S[gfrom, gto])

        b = next(i for i, s in enumerate(lines) if "SIGMA_T_BEGIN" in s)
        e = next(i for i, s in enumerate(lines) if "SIGMA_T_END" in s)
        for i in range(b + 1, e):
            toks = lines[i].split()
            g = int(toks[0])
            toks[1] = "{:.12g}".format(float(sigma_t_vec[g]))
            lines[i] = " ".join(toks) + "\n"

        tb = next(i for i, s in enumerate(lines) if "TRANSFER_MOMENTS_BEGIN" in s)
        te = next(i for i, s in enumerate(lines) if "TRANSFER_MOMENTS_END" in s)

        new_tm = []
        for gfrom in range(G):
            for gto in range(G):
                val = float(S[gfrom, gto])
                if abs(val) > self.transfer_tol:
                    new_tm.append(
                        "M_GFROM_GTO_VAL 0 {} {} {:.12g}\n".format(gfrom, gto, val)
                    )

        lines[tb + 1:te] = new_tm

        with open(out_file, "w") as f:
            f.writelines(lines)

    def _update_fission_xs(self, in_file, out_file, sigma_f_vec, sigma_c_vec, S):
        with open(in_file, "r") as f:
            lines = f.readlines()

        sigma_t_vec = np.zeros_like(sigma_f_vec, dtype=float)

        b = next(i for i, s in enumerate(lines) if "SIGMA_F_BEGIN" in s)
        e = next(i for i, s in enumerate(lines) if "SIGMA_F_END" in s)
        for i in range(b + 1, e):
            toks = lines[i].split()
            g = int(toks[0])
            toks[1] = "{:.12g}".format(float(sigma_f_vec[g]))
            lines[i] = " ".join(toks) + "\n"
            sigma_t_vec[g] = float(sigma_f_vec[g]) + float(sigma_c_vec[g])

        tb = next(i for i, s in enumerate(lines) if "TRANSFER_MOMENTS_BEGIN" in s)
        te = next(i for i, s in enumerate(lines) if "TRANSFER_MOMENTS_END" in s)

        G = len(sigma_t_vec)
        new_tm = []
        for gfrom in range(G):
            for gto in range(G):
                val = float(S[gfrom, gto])
                if abs(val) > self.transfer_tol:
                    new_tm.append(
                        "M_GFROM_GTO_VAL 0 {} {} {:.12g}\n".format(gfrom, gto, val)
                    )
                sigma_t_vec[gfrom] += val

        lines[tb + 1:te] = new_tm

        b = next(i for i, s in enumerate(lines) if "SIGMA_T_BEGIN" in s)
        e = next(i for i, s in enumerate(lines) if "SIGMA_T_END" in s)
        for i in range(b + 1, e):
            toks = lines[i].split()
            g = int(toks[0])
            toks[1] = "{:.12g}".format(float(sigma_t_vec[g]))
            lines[i] = " ".join(toks) + "\n"

        with open(out_file, "w") as f:
            f.writelines(lines)

    # -------------------------------------------------------------------------
    # Readers
    # -------------------------------------------------------------------------

    def _read_lines(self, xs_file):
        with open(xs_file, "r") as f:
            return f.readlines()

    def _find_block(self, lines, begin_marker, end_marker):
        b = next(i for i, s in enumerate(lines) if begin_marker in s)
        e = next(i for i, s in enumerate(lines) if end_marker in s)
        return b, e

    def _read_num_groups(self, lines):
        line = next(s for s in lines if s.strip().startswith("NUM_GROUPS"))
        return int(line.split()[1])

    def _read_vector_block(self, lines, begin_marker, end_marker):
        G = self._read_num_groups(lines)
        vec = np.zeros(G, dtype=float)

        b, e = self._find_block(lines, begin_marker, end_marker)
        for line in lines[b + 1:e]:
            toks = line.split()
            if len(toks) < 2:
                continue
            g = int(toks[0])
            vec[g] = float(toks[1])

        return vec

    def _read_transfer_entries(self, lines):
        entries = []
        values = []

        b, e = self._find_block(lines, "TRANSFER_MOMENTS_BEGIN", "TRANSFER_MOMENTS_END")
        for line in lines[b + 1:e]:
            toks = line.split()

            if len(toks) < 5:
                continue

            key = toks[0]
            if key not in ("M_GFROM_GTO_VAL", "M_GPRIME_G_VAL"):
                continue

            ell = int(toks[1])
            if ell != 0:
                continue

            gfrom = int(toks[2])
            gto = int(toks[3])
            val = float(toks[4])

            entries.append((gfrom, gto))
            values.append(val)

        return entries, np.array(values, dtype=float)

    def _entries_to_matrix(self, G, entries, values):
        S = np.zeros((G, G), dtype=float)
        for (gfrom, gto), val in zip(entries, values):
            S[gfrom, gto] = val
        return S

    def _read_total_xs(self, xs_file):
        lines = self._read_lines(xs_file)
        G = self._read_num_groups(lines)

        sigma_t_vec = self._read_vector_block(lines, "SIGMA_T_BEGIN", "SIGMA_T_END")
        transfer_entries, transfer_vals = self._read_transfer_entries(lines)

        S = self._entries_to_matrix(G, transfer_entries, transfer_vals)
        sigma_a_vec = sigma_t_vec - S.sum(axis=1)

        return {
            "kind": "total",
            "file": Path(xs_file),
            "G": G,
            "sigma_t_vec": sigma_t_vec,
            "sigma_a_vec": sigma_a_vec,
            "transfer_entries": transfer_entries,
            "transfer_vals": transfer_vals,
        }

    def _read_fission_xs(self, xs_file):
        lines = self._read_lines(xs_file)
        G = self._read_num_groups(lines)

        sigma_t_vec = self._read_vector_block(lines, "SIGMA_T_BEGIN", "SIGMA_T_END")
        sigma_f_vec = self._read_vector_block(lines, "SIGMA_F_BEGIN", "SIGMA_F_END")
        transfer_entries, transfer_vals = self._read_transfer_entries(lines)

        S = self._entries_to_matrix(G, transfer_entries, transfer_vals)
        sigma_c_vec = sigma_t_vec - sigma_f_vec - S.sum(axis=1)

        return {
            "kind": "fission",
            "file": Path(xs_file),
            "G": G,
            "sigma_t_vec": sigma_t_vec,
            "sigma_f_vec": sigma_f_vec,
            "sigma_c_vec": sigma_c_vec,
            "transfer_entries": transfer_entries,
            "transfer_vals": transfer_vals,
        }

    # -------------------------------------------------------------------------
    # Flatten / unflatten: entrywise
    # -------------------------------------------------------------------------

    def _flatten_total_data(self, data):
        return np.concatenate([data["sigma_a_vec"], data["transfer_vals"]])

    def _flatten_fission_data(self, data):
        return np.concatenate([data["sigma_f_vec"], data["sigma_c_vec"], data["transfer_vals"]])

    def _unflatten_total_data(self, x_block, block):
        G = block["G"]
        n_transfer = len(block["transfer_entries"])

        sigma_a_vec = np.array(x_block[:G], dtype=float)
        transfer_vals = np.array(x_block[G:G + n_transfer], dtype=float)
        S = self._entries_to_matrix(G, block["transfer_entries"], transfer_vals)

        return sigma_a_vec, S

    def _unflatten_fission_data(self, x_block, block):
        G = block["G"]
        n_transfer = len(block["transfer_entries"])

        i0 = 0
        i1 = i0 + G
        i2 = i1 + G
        i3 = i2 + n_transfer

        sigma_f_vec = np.array(x_block[i0:i1], dtype=float)
        sigma_c_vec = np.array(x_block[i1:i2], dtype=float)
        transfer_vals = np.array(x_block[i2:i3], dtype=float)
        S = self._entries_to_matrix(G, block["transfer_entries"], transfer_vals)

        return sigma_f_vec, sigma_c_vec, S

    # -------------------------------------------------------------------------
    # Flatten / unflatten: block scales
    # -------------------------------------------------------------------------

    def _flatten_total_scales(self, data):
        return np.array([1.0, 1.0], dtype=float)

    def _flatten_fission_scales(self, data):
        return np.array([1.0, 1.0, 1.0], dtype=float)

    def _unflatten_total_scales(self, x_block, block):
        data = block["data"]

        sigma_a_scale = float(x_block[0])
        scatter_scale = float(x_block[1])

        sigma_a_vec = sigma_a_scale * np.asarray(data["sigma_a_vec"], dtype=float)
        transfer_vals = scatter_scale * np.asarray(data["transfer_vals"], dtype=float)
        S = self._entries_to_matrix(block["G"], block["transfer_entries"], transfer_vals)

        return sigma_a_vec, S

    def _unflatten_fission_scales(self, x_block, block):
        data = block["data"]

        sigma_f_scale = float(x_block[0])
        sigma_c_scale = float(x_block[1])
        scatter_scale = float(x_block[2])

        sigma_f_vec = sigma_f_scale * np.asarray(data["sigma_f_vec"], dtype=float)
        sigma_c_vec = sigma_c_scale * np.asarray(data["sigma_c_vec"], dtype=float)
        transfer_vals = scatter_scale * np.asarray(data["transfer_vals"], dtype=float)
        S = self._entries_to_matrix(block["G"], block["transfer_entries"], transfer_vals)

        return sigma_f_vec, sigma_c_vec, S

    # -------------------------------------------------------------------------
    # Bounds
    # -------------------------------------------------------------------------

    def _relative_bounds(self, x_nominal):
        x_nominal = np.asarray(x_nominal, dtype=float)
        a = (1.0 - self.frac) * x_nominal
        b = (1.0 + self.frac) * x_nominal
        lower = np.minimum(a, b)
        upper = np.maximum(a, b)
        return list(zip(lower, upper))