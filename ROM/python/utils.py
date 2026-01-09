import numpy as np
import subprocess
import h5py
from scipy.stats import qmc


def load_2d_flux(file_pattern, ranks, moment=0):
    """Load (x, y, flux) grouped by energy group from HDF5 files."""
    with h5py.File(file_pattern.format(ranks[0]), "r") as f0:
        num_groups = int(f0.attrs["num_groups"])
        num_moments = int(f0.attrs["num_moments"])

    xs = [[] for _ in range(num_groups)]
    ys = [[] for _ in range(num_groups)]
    vals = [[] for _ in range(num_groups)]

    for r in ranks:
        fp = file_pattern.format(r)
        with h5py.File(fp, "r") as f:
            x = f["mesh/nodes_x"][:]
            y = f["mesh/nodes_y"][:]
            values = f["values"][:]
            num_nodes = x.size

        values = values.reshape(num_nodes, num_moments, num_groups)
        for g in range(num_groups):
            xs[g].append(x)
            ys[g].append(y)
            vals[g].append(values[:, moment, g])

    for g in range(num_groups):
        xs[g] = np.concatenate(xs[g])
        ys[g] = np.concatenate(ys[g])
        vals[g] = np.concatenate(vals[g])

    return xs, ys, vals, num_groups

def load_1d_flux(file_pattern, ranks, moment=0):
    """Load concatenated 1-D (x, flux) data per energy group."""
    with h5py.File(file_pattern.format(ranks[0]), "r") as f0:
        num_groups = int(f0.attrs["num_groups"])
        num_moments = int(f0.attrs["num_moments"])

    xs = [[] for _ in range(num_groups)]
    vals = [[] for _ in range(num_groups)]

    for r in ranks:
        fp = file_pattern.format(r)
        with h5py.File(fp, "r") as f:
            x = f["mesh/nodes_z"][:]
            values = f["values"][:]
            num_nodes = x.size

        values = values.reshape(num_nodes, num_moments, num_groups)

        for g in range(num_groups):
            xs[g].append(x)
            vals[g].append(values[:, moment, g])

    for g in range(num_groups):
        xs[g] = np.concatenate(xs[g])
        vals[g] = np.concatenate(vals[g])

    return xs, vals, num_groups


def sample_parameter_space(bounds, n_samples):
    """Performs uniform sampling on a set of bounds and includes the vertices of the parameter domain"""
    n_dim = len(bounds)
    n_vertices = 2**n_dim
    n_random = n_samples - n_vertices

    # Random interior samples
    random_samples = np.array([
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(n_random)
    ])

    # Vertices of domain
    vertices = np.zeros((n_vertices, n_dim))
    for i in range(n_vertices):
        for d, (low, high) in enumerate(bounds):
            if (i >> d) & 1:
                vertices[i, d] = high
            else:
                vertices[i, d] = low

    samples = np.vstack([random_samples, vertices])
    return samples

def sample_LHS(bounds, n_samples):
    """Performs Latin Hypercube Sampling on a set of bounds"""
    d = len(bounds)

    # Create LHS sampler
    sampler = qmc.LatinHypercube(d=d)

    # Generate samples in [0, 1]^d
    unit_samples = sampler.random(n=n_samples)

    # Scale to physical bounds
    lows  = np.array([low for (low, high) in bounds])
    highs = np.array([high for (low, high) in bounds])

    samples = qmc.scale(unit_samples, lows, highs)
    return samples

def sample_test(bounds, n_samples):
    """Performs uniform sampling on a set of bounds to generate a test set"""
    samples = np.array([
        [np.random.uniform(low, high) for (low, high) in bounds]
        for _ in range(n_samples)
    ])

    return samples
