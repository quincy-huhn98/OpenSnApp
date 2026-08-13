import numpy as np
from scipy.stats import qmc


class ActiveSubspace:
    def __init__(self, bounds):
        """
        bounds: array-like of shape (n_params, 2)
                Each row is [lower, upper].
        """
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_params = self.bounds.shape[0]

        self.gradients = None
        self.G = None
        self.evals = None
        self.evecs = None
        self.W_active = None
        self.rank = None

    def add_gradients(self, gradients):
        """
        gradients: array-like of shape (n_samples, n_params)

        Gradients are assumed to be with respect to physical parameters.
        They are converted to gradients with respect to normalized parameters.
        """
        gradients = np.asarray(gradients, dtype=float)

        if gradients.ndim != 2:
            raise ValueError("gradients must have shape (n_samples, n_params).")

        if gradients.shape[1] != self.n_params:
            raise ValueError(
                f"Expected gradients with {self.n_params} parameters, "
                f"got {gradients.shape[1]}."
            )

        scale = 0.5 * (self.bounds[:, 1] - self.bounds[:, 0])
        gradients_normalized = gradients * scale[None, :]

        # Store as G = [grad f_1, ..., grad f_N]
        self.gradients = gradients_normalized.T

    def compute_subspace(self):
        """
        Assemble G = [grad f_1, ..., grad f_N] / sqrt(N) and compute its SVD.

        This keeps the current SVD-based implementation.
        """
        if self.gradients is None:
            raise RuntimeError("No gradients have been added.")

        n_samples = self.gradients.shape[1]

        self.G = self.gradients / np.sqrt(n_samples)

        W, Lambda, _ = np.linalg.svd(self.G)

        self.evals = Lambda
        self.evecs = W

        np.savetxt("results/AS_values.txt", self.evals)
        np.savetxt("results/AS_vectors.txt", self.evecs)

    def set_rank(self, rank):
        """
        Select the number of active dimensions.
        """
        if self.evecs is None:
            raise RuntimeError("Call compute_subspace() first.")

        if rank < 1 or rank > self.n_params:
            raise ValueError("rank must be between 1 and n_params.")

        self.rank = rank
        self.W_active = self.evecs[:, :rank]

    def normalize(self, theta):
        """
        Map physical parameters to [-1, 1]^d.
        """
        theta = np.asarray(theta, dtype=float)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]

        return 2.0 * (theta - lower) / (upper - lower) - 1.0

    def unnormalize(self, x):
        """
        Map normalized parameters from [-1, 1]^d to physical parameters.
        """
        x = np.asarray(x, dtype=float)
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]

        return lower + 0.5 * (x + 1.0) * (upper - lower)

    def project_to_active(self, theta):
        """
        Project physical parameters onto the active subspace.

        Uses the literature convention

            y = W_active^T x

        with x stored internally as column samples.

        Returns
        -------
        y : ndarray, shape (n_samples, rank)
            Row-wise active coordinates expected by rom_driver.py.
        """
        if self.W_active is None:
            raise RuntimeError("Call set_rank(rank) first.")

        x = self.normalize(theta)

        if x.ndim == 1:
            x = x[None, :]

        y = self.W_active.T @ x.T

        return y.T

    def reconstruct_from_active(self, y, inactive_value=None):
        """
        Map active coordinates back to full physical parameter space.

        Uses the literature convention

            x = W_active y + W_inactive z

        with y and z stored internally as column samples.
        """
        if self.W_active is None:
            raise RuntimeError("Call set_rank(rank) first.")

        y = np.asarray(y, dtype=float)

        if y.ndim == 1:
            y = y[None, :]

        if y.shape[1] != self.rank:
            raise ValueError(f"Expected active samples with dimension {self.rank}.")

        W1 = self.W_active
        W2 = self.evecs[:, self.rank:]

        y_col = y.T

        if inactive_value is None:
            x_col = W1 @ y_col
        else:
            inactive_value = np.asarray(inactive_value, dtype=float)

            if inactive_value.ndim == 1:
                inactive_value = inactive_value[None, :]

            inactive_dim = self.n_params - self.rank
            if inactive_value.shape[1] != inactive_dim:
                raise ValueError(
                    f"Expected inactive samples with dimension {inactive_dim}."
                )

            x_col = W1 @ y_col + W2 @ inactive_value.T

        return self.unnormalize(x_col.T)

    def make_active_training_set(
        self,
        n_samples,
        method="lhs",
        inactive_scale=0.0,
        clip=True,
        reject_outside=False,
        max_attempts=10000,
    ):
        """
        Create a training set biased by the active subspace.

        Samples are generated in normalized coordinates as

            x = W_active y + W_inactive z

        where y is sampled in the active variables and z is sampled in the
        inactive variables. If inactive_scale=0.0, this recovers the original
        active-only behavior.

        Returns
        -------
        theta_train : ndarray, shape (n_samples, n_params)
            Physical parameter samples for OMMI training.
        y_train : ndarray, shape (n_samples, rank)
            Corresponding active variables.
        x_train : ndarray, shape (n_samples, n_params)
            Normalized full-space samples.
        """
        if self.W_active is None:
            raise RuntimeError("Call set_rank(rank) first.")

        if n_samples < 1:
            raise ValueError("n_samples must be positive.")

        if inactive_scale < 0.0 or inactive_scale > 1.0:
            raise ValueError("inactive_scale must be between 0 and 1.")

        W1 = self.W_active
        W2 = self.evecs[:, self.rank:]
        inactive_dim = self.n_params - self.rank

        def sample_unit(n, dim):
            if dim == 0:
                return np.zeros((n, 0))

            if method == "lhs":
                sampler = qmc.LatinHypercube(d=dim)
                u = sampler.random(n)
                return 2.0 * u - 1.0

            if method == "uniform":
                return np.random.uniform(-1.0, 1.0, size=(n, dim))

            raise ValueError("method must be 'lhs' or 'uniform'.")

        accepted_x = []
        attempts = 0
        n_collected = 0

        while n_collected < n_samples:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Only generated {n_collected}/{n_samples} valid samples. "
                    "Try smaller inactive_scale or disable rejection."
                )

            n_needed = n_samples - n_collected
            n_batch = max(5 * n_needed, 50)

            y = sample_unit(n_batch, self.rank)

            if inactive_dim > 0 and inactive_scale > 0.0:
                z = inactive_scale * sample_unit(n_batch, inactive_dim)
                x = W1 @ y.T + W2 @ z.T
            else:
                x = W1 @ y.T

            if reject_outside:
                valid = np.all((x >= -1.0) & (x <= 1.0), axis=0)
                x = x[:, valid]
            elif clip:
                x = np.clip(x, -1.0, 1.0)

            if x.shape[1] > 0:
                accepted_x.append(x)
                n_collected += x.shape[1]

        x_all = np.hstack(accepted_x)[:, :n_samples]
        x_train = x_all.T

        # Recompute accepted active coordinates using y = W_active^T x.
        y_train = (self.W_active.T @ x_all).T

        theta_train = self.unnormalize(x_train)

        return theta_train, y_train, x_train