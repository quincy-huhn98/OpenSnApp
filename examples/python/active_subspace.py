from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Literal, Protocol, Callable, Any
import numpy as np
#from scipy.stats import qmc


# -----------------------------
# Common types
# -----------------------------

Status = Literal["ok", "failed", "timeout", "parse_error", "nan"]


@dataclass(frozen=True)
class EvalRequest:
    x: np.ndarray
    z: np.ndarray
    stage: str
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalResult:
    ok: bool
    status: Status
    f: Optional[float]
    runtime_s: float
    run_dir: Optional[str] = None
    stderr_tail: Optional[str] = None
    stdout_tail: Optional[str] = None
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SubspaceBasis:
    W1: np.ndarray
    lambdas: np.ndarray
    U: Optional[np.ndarray] = None
    S: Optional[np.ndarray] = None
    Vt: Optional[np.ndarray] = None
    n_active: int = 0


@dataclass(frozen=True)
class SubspaceOptions:
    energy_threshold: float = 0.99
    min_active: int = 1
    max_active: Optional[int] = None

"""# Alg 2.2"""

# -----------------------------
# Adaptive Morris (Alg. 2.2)
# -----------------------------

@dataclass(frozen=True)
class AdaptiveMorrisOptions:
    M_max: int = 200
    delta_min: float = 0.01
    delta_max: float = 0.05
    energy_threshold: float = 0.99
    check_every: int = 10
    patience: int = 3

    # NEW: basis convergence
    basis_tol: float = 5e-2          # tune; try 1e-2 to 5e-2 for deterministic mock
    basis_k: Optional[int] = None    # compare first k vectors; default uses current n_active
    basis_metric: str = "proj_fro"   # "proj_fro" or "max_angle"
    basis_warmup: int = 0
    finite_diff_eps: float = 0.0  # if >0, adds a small denom stabilization


@dataclass(frozen=True)
class ColumnPlan:
    column_index: int
    z0: np.ndarray
    f0: float
    n_trunc: int
    deltas: np.ndarray          # (n_trunc,)
    directions: np.ndarray      # (m, r_used) - full direction basis used to map g_tilde -> g


@dataclass
class AdaptiveMorrisState:
    m: int
    G_cols: List[np.ndarray] = field(default_factory=list)   # each (m,)
    basis: Optional[SubspaceBasis] = None

    iterations: int = 0
    history: List[Dict[str, object]] = field(default_factory=list)
    last_checked_W1: Optional[np.ndarray] = None
    stable_checks_basis: int = 0
    checks_done: int = 0


def _energy_truncation_index(lambdas, tau, min_active, max_active=None):
    if lambdas.size == 0 or float(np.sum(lambdas)) <= 0.0:
        n = min_active
    else:
        total = float(np.sum(lambdas))
        cum = np.cumsum(lambdas) / total
        n = int(np.searchsorted(cum, tau) + 1)
        n = max(n, min_active)
    if max_active is not None:
        n = min(n, max_active)
    return n


def _rand_uniform_cube(rng, m):
    return rng.uniform(-1.0, 1.0, size=(m,))


def _safe_eval(evaluator, req):
    res = evaluator(req.x)
    return res

def _subspace_err_proj_fro(W_old: np.ndarray, W_new: np.ndarray) -> float:
    # W_*: (m,n) with orthonormal cols (as returned by SVD)
    P_old = W_old @ W_old.T
    P_new = W_new @ W_new.T
    return float(np.linalg.norm(P_new - P_old, ord="fro"))

def _subspace_err_max_angle(W_old: np.ndarray, W_new: np.ndarray) -> float:
    # principal angles via svd of n×n matrix
    M = W_old.T @ W_new
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta_max = float(np.arccos(np.min(s)))
    return theta_max


class AdaptiveMorris:
    """
    Concrete, robust implementation of Adaptive Morris (Alg. 2.2).

    Notes:
    - Works entirely in normalized z-space for stepping; uses normalizer to map to x for evaluation.
    - Uses current SVD directions from G (via subspace.compute) to choose step directions.
    - When M < m, SVD returns U with r=min(m, M). This is fine: we only step along available r directions.
    """

    def alg21_first_column(
        self,
        *,
        evaluator,
        normalizer,
        rng,
        opts: AdaptiveMorrisOptions,
        m: int,
        M_scale: int,
        coords: Optional[np.ndarray] = None,
    ):
        """
        Algorithm 2.1 (Finite-Difference Morris) column.

        Paper:
          - pick base point z*
          - choose D_i = ±Δ (random sign per component)
          - g_i = (f(z* + D_i e_i) - f(z*)) / D_i
          - G(:,j) = (1/sqrt(M)) g

        Engineering:
          - coords: optional subset of indices to probe (sparse Morris) for large m
          - Δ: uses mid of [delta_min, delta_max] by default
        """
        # Choose Δ for Alg. 2.1 (paper uses a single Δ)
        Delta = 0.02# 0.5 * (opts.delta_min + opts.delta_max)
        Delta = float(Delta)
        if Delta <= 0.0:
            raise ValueError("Alg2.1 Delta must be positive; check delta_min/delta_max.")

        if coords is None:
            coords = np.arange(m, dtype=int)
        else:
            coords = np.asarray(coords, dtype=int)
            if coords.ndim != 1:
                raise ValueError("coords must be 1D.")
            if np.any(coords < 0) or np.any(coords >= m):
                raise ValueError("coords contains out-of-range indices.")

        # Base point z*
        z0 = _rand_uniform_cube(rng, m)
        x0 = normalizer.z_to_x(z0)
        f0 = _safe_eval(evaluator, EvalRequest(x=x0, z=z0, stage="alg21_base", meta={"alg": "2.1"}))

        # Random signs D_i = ±Δ on probed coords
        signs = rng.choice(np.array([-1.0, 1.0], dtype=float), size=(coords.size,))
        D = Delta * signs

        g = np.zeros((m,), dtype=float)

        for k, i in enumerate(coords):
            di = float(D[k])

            z1 = z0.copy()
            z1[i] += di

            x1 = normalizer.z_to_x(z1)
            f1 = _safe_eval(
                evaluator,
                EvalRequest(
                    x=x1,
                    z=z1,
                    stage="alg21_step",
                    meta={"alg": "2.1", "coord": int(i), "D": float(di), "Delta": float(Delta)},
                ),
            )

            denom = di + float(opts.finite_diff_eps)
            if abs(denom) <= 0.0:
                raise ValueError("Alg2.1 denominator is zero; check finite_diff_eps.")
            g[i] = (f1 - f0) / denom

        # Paper scaling: 1/sqrt(M)
        g_col = (1.0 / float(np.sqrt(max(1, int(M_scale))))) * g

        meta = {
            "z0": z0.tolist(),
            "f0": float(f0),
            "Delta": float(Delta),
            "coords_count": int(coords.size),
            "M_scale": int(M_scale),
        }

        return g_col, meta


    def run(self, *, evaluator, normalizer, subspace, x_nominal, rng, opts, step_policy, subspace_opts, initial_G=None):

        x_nominal = np.asarray(x_nominal, dtype=float)
        m = int(x_nominal.size)
        state = AdaptiveMorrisState(m=m)

        # Seed basis if provided (typically from initialization stage)
        if initial_G is not None:
            G0 = np.zeros((m, len(initial_G)), dtype=float)
            for j, d in enumerate(initial_G):
              G0[:, j] = np.asarray(d.v, dtype=float)
            if G0.ndim != 2 or G0.shape[0] != m:
                raise ValueError(f"initial_G must have shape (m,k), got {G0.shape}")
            # split into columns
            for j in range(G0.shape[1]):
                state.G_cols.append(G0[:, j].copy())
            G = np.column_stack(state.G_cols)
            basis = subspace.compute(G, SubspaceOptions(
                energy_threshold=subspace_opts.energy_threshold,
                min_active=subspace_opts.min_active,
                max_active=subspace_opts.max_active,
            ))
            state.basis = basis
        else:
            # For large m, you probably want sparse coords (e.g., 20–50) instead of all m.
            # For your ridge mock test, leave coords=None to use all coords and verify angle improvement.
            coords = None
            # Example sparse option:
            # coords = rng.choice(m, size=min(30, m), replace=False)

            g0, meta0 = self.alg21_first_column(
                evaluator=evaluator,
                normalizer=normalizer,
                rng=rng,
                opts=opts,
                m=m,
                M_scale=opts.M_max,   # paper uses M (desired number of columns) in scaling
                coords=coords,
            )
            state.G_cols.append(g0)
            state.iterations += 1

            # Update basis immediately from the seeded column so the loop starts with an SVD-defined U
            G = np.column_stack(state.G_cols)  # (m,1)
            basis = subspace.compute(
                G,
                SubspaceOptions(
                    energy_threshold=subspace_opts.energy_threshold,
                    min_active=subspace_opts.min_active,
                    max_active=subspace_opts.max_active,
                ),
            )
            state.basis = basis

            state.history.append({
                "iter": state.iterations,
                "M": len(state.G_cols),
                "n_active": basis.n_active,
                "lambdas_head": basis.lambdas[: min(10, basis.lambdas.size)].tolist(),
                "plan_n_trunc": None,
                "plan_z0": meta0["z0"],
                "plan_f0": meta0["f0"],
                "alg21_Delta": meta0["Delta"],
                "alg21_coords_count": meta0["coords_count"],
            })

        for j in range(opts.M_max):
            plan = self.plan_column(
                state=state,
                rng=rng,
                opts=opts,
                subspace_opts=subspace_opts,
                step_policy=step_policy,
                normalizer=normalizer,
                evaluator=evaluator,
                x_nominal=x_nominal,
            )

            g_col = self.evaluate_column(
                plan=plan,
                evaluator=evaluator,
                normalizer=normalizer,
                opts=opts,
                rng=rng
            )
            state.G_cols.append(g_col)
            state.iterations += 1

            # Update basis from all collected columns
            G = np.column_stack(state.G_cols)  # (m, M_so_far)
            basis = subspace.compute(G, SubspaceOptions(
                energy_threshold=subspace_opts.energy_threshold,
                min_active=subspace_opts.min_active,
                max_active=subspace_opts.max_active,
            ))
            state.basis = basis

            # History snapshot
            state.history.append({
                "iter": state.iterations,
                "M": len(state.G_cols),
                "n_active": basis.n_active,
                "lambdas_head": basis.lambdas[: min(10, basis.lambdas.size)].tolist(),
                "plan_n_trunc": int(plan.n_trunc),
                "plan_z0": plan.z0.tolist(),
                "plan_f0": float(plan.f0),
            })

            # Convergence checks
            if (opts.check_every > 0) and (state.iterations % opts.check_every == 0):
                if self.check_convergence(state=state, opts=opts):
                    break

        if state.basis is None:
            # This can only happen if M_max == 0
            raise RuntimeError("AdaptiveMorris produced no basis; check M_max.")

        return state, state.basis

    def plan_column(self, *, state, rng, opts, subspace_opts, step_policy, normalizer, evaluator, x_nominal):

        m = state.m

        # Choose a base point z0 in [-1,1]^m
        # (You can swap this sampler later for LHS or a biased sampler.)
        # NEW (interior sampling):
        margin = float(opts.delta_max)  # ensures z0 +/- di*ui likely stays in bounds
        hi = 1.0 - margin
        lo = -hi
        z0 = rng.uniform(lo, hi, size=m)

        x0 = normalizer.z_to_x(z0)
        f0 = _safe_eval(evaluator, EvalRequest(x=x0, z=z0, stage="grad_base", meta={"col": len(state.G_cols)}))

        # Choose direction basis to step in:
        # Prefer the most recent SVD U, else initial_U if present, else identity.
        U = None
        lambdas = np.array([], dtype=float)

        if state.basis is not None and state.basis.U is not None:
            U = np.asarray(state.basis.U, dtype=float)
            lambdas = np.asarray(state.basis.lambdas, dtype=float)
        else:
            U = np.eye(m, dtype=float)
            lambdas = np.ones((m,), dtype=float)  # neutral until we have information

        # Determine truncation n_trunc: energy threshold on current lambdas
        # but limited by available directions in U
        n_energy = _energy_truncation_index(
            lambdas=lambdas,
            tau=opts.energy_threshold,
            min_active=subspace_opts.min_active,
            max_active=subspace_opts.max_active,
        )
        r = int(U.shape[1])
        n_floor = int(min(r, subspace_opts.min_active))
        n_cap   = int(min(r, subspace_opts.max_active))

        M_so_far = len(state.G_cols)  # columns already in G (before adding this one)

        # 1) Burn-in: don't trust eigenvalues yet
        M_burn = max(5, 2 * n_floor)
        if M_so_far < M_burn or lambdas.size == 0:
            n_trunc = n_floor
        else:
            # 2) Energy-based choice (bounded)
            n_energy = _energy_truncation_index(
                lambdas=lambdas,
                tau=opts.energy_threshold,
                min_active=subspace_opts.min_active,
                max_active=subspace_opts.max_active,
            )
            n_energy = int(np.clip(n_energy, n_floor, n_cap))

            # 3) Hysteresis: don't allow shrinking too aggressively
            # (prevents "collapse to 1 forever" from one noisy spectrum)
            prev = getattr(state, "last_n_trunc", None)
            if prev is None:
                n_trunc = n_energy
            else:
                # allow growth immediately; allow shrink by at most 1 per iteration
                if n_energy >= prev:
                    n_trunc = n_energy
                else:
                    n_trunc = max(n_energy, prev - 1)

        # stash for next iteration
        state.last_n_trunc = int(n_trunc)

        # Direction-dependent step sizes
        deltas = np.array([step_policy.delta(lambdas, i, opts) for i in range(n_trunc)], dtype=float)
        deltas = np.clip(deltas, opts.delta_min, opts.delta_max)

        return ColumnPlan(
            column_index=len(state.G_cols),
            z0=z0,
            f0=f0,
            n_trunc=n_trunc,
            deltas=deltas,
            directions=U,  # full basis used for mapping g_tilde -> g
        )

    def evaluate_column(self, *, plan, evaluator, normalizer, opts, rng):
        z0 = np.asarray(plan.z0, dtype=float)
        f0 = float(plan.f0)
        U = np.asarray(plan.directions, dtype=float)  # (m, r)
        m, r = U.shape
        n_trunc = int(plan.n_trunc)
        deltas = np.asarray(plan.deltas, dtype=float)

        g_tilde = np.zeros((r,), dtype=float)

        for i in range(n_trunc):
            ui = U[:, i]
            di = float(deltas[i])

            sgn = float(rng.choice([-1.0, 1.0]))
            z1 = z0 + sgn * di * ui

            x1 = normalizer.z_to_x(z1)
            f1 = _safe_eval(
                evaluator,
                EvalRequest(
                    x=x1,
                    z=z1,
                    stage="grad_step",
                    meta={"col": int(plan.column_index), "dir": int(i), "delta": float(di)},
                ),
            )

            denom = di
            if denom <= 0.0:
                raise ValueError("Non-positive denominator in finite difference; check delta/finite_diff_eps.")

            g_tilde[i] = (f1 - f0) / denom

        g_col = U @ g_tilde
        return np.asarray(g_col, dtype=float).reshape((m,))


    def check_convergence(self, *, state, opts):
        basis = state.basis
        if basis is None or basis.W1 is None:
            return False

        W1 = np.asarray(basis.W1, dtype=float)
        m, n_cur = W1.shape
        if n_cur == 0:
            return False

        # choose k for this check
        k = int(n_cur if opts.basis_k is None else min(opts.basis_k, n_cur))
        if k <= 0:
            return False

        W_new = W1[:, :k]

        # bookkeeping
        state.checks_done += 1
        if state.checks_done <= int(opts.basis_warmup):
            state.last_checked_W1 = W_new.copy()
            state.stable_checks_basis = 0
            return False

        W_old = state.last_checked_W1
        if W_old is None:
            state.last_checked_W1 = W_new.copy()
            state.stable_checks_basis = 0
            return False

        k_old = int(W_old.shape[1])

        # NEW RULE: only check when basis dimension matches previous
        if k_old != k:
            # dimension changed -> reset stability counter and refresh cache
            state.last_checked_W1 = W_new.copy()
            state.stable_checks_basis = 0

            if state.history:
                state.history[-1]["basis_err"] = None
                state.history[-1]["basis_k"] = int(k)
                state.history[-1]["basis_metric"] = str(opts.basis_metric)
                state.history[-1]["basis_dim_changed"] = True
            return False

        # dimensions match: compute subspace change
        if opts.basis_metric == "max_angle":
            err = _subspace_err_max_angle(W_old, W_new)  # radians
            ok = (err < float(opts.basis_tol))
        else:
            err = _subspace_err_proj_fro(W_old, W_new)
            # optional normalization (keeps scale ~O(1)):
            # err = err / np.sqrt(2.0 * k)
            ok = (err < float(opts.basis_tol))

        if ok:
            state.stable_checks_basis += 1
        else:
            state.stable_checks_basis = 0

        state.last_checked_W1 = W_new.copy()

        if state.history:
            state.history[-1]["basis_err"] = float(err)
            state.history[-1]["basis_k"] = int(k)
            state.history[-1]["basis_metric"] = str(opts.basis_metric)
            state.history[-1]["basis_dim_changed"] = False

        return state.stable_checks_basis >= int(opts.patience)

"""# ALG 4.1"""

def _rand_unit_vector(rng, m):
    v = rng.normal(size=m)
    n = float(np.linalg.norm(v))
    if n <= 1e-15:
        v = np.zeros(m)
        v[0] = 1.0
        return v
    return v / n

def _project_to_sphere_about_center(z, z0, delta) :
    d = z - z0
    nd = float(np.linalg.norm(d))
    if nd <= 1e-15:
        # choose an arbitrary direction if we collapsed
        d = np.zeros_like(d)
        d[0] = 1.0
        nd = 1.0
    return z0 + (delta / nd) * d

def _sample_on_sphere_about_center(rng, z0, delta):
    """
    Sample z on sphere ||z - z0|| = delta.
    """
    z0 = np.asarray(z0, float)
    m = z0.size

    u = _rand_unit_vector(rng, m)
    z = z0 + delta * u
    z = _project_to_sphere_about_center(z, z0, delta)
    return z

# -----------------------------
# Initialization (Alg. 4.1-like)
# -----------------------------

@dataclass(frozen=True)
class InitGreatCircleOptions:
    N_init: int = 50
    ell: int = 60
    delta_init: float = 0.10


@dataclass(frozen=True)
class InitDirection:
    z0: np.ndarray
    z_best: np.ndarray
    v: np.ndarray
    f0: float
    f_best: float
    evals_used: int
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class InitState:
    m: int
    directions: List[InitDirection] = field(default_factory=list)
    history: List[Dict[str, object]] = field(default_factory=list)


def _rand_unit_vector(rng, m):
    v = rng.normal(0.0, 1.0, size=(m,))
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        v[0] = 1.0
        nrm = 1.0
    return v / nrm


class InitGreatCircle:
    """
    Engineering-robust initialization of Alg. 4.1.
    """

    def run(self, *, evaluator, normalizer, rng, x_nominal, opts, subspace, subspace_opts, energy_threshold=0.99, check_every=1):

        x_nominal = np.asarray(x_nominal, dtype=float)
        m = int(x_nominal.size)
        state = InitState(m=m)

        max_cols = int(opts.N_init)  # treat opts.N_init as a hard cap now

        # We'll accumulate V incrementally so SVD checks are easy.
        V = np.zeros((m, max_cols), dtype=float)

        # Bounds for "how many directions we will keep" when we terminate
        min_active = int(getattr(subspace_opts, "min_active", 1) or 1)
        max_active = getattr(subspace_opts, "max_active", None)
        max_active = int(max_active) if max_active is not None else None

        # Best-so-far outputs
        U_best = None
        n_keep_best = None

        for k in range(max_cols):
            # Base point z0 in [-1,1]^m, keep margin so sphere stays inside cube
            z0 = rng.uniform(-1.0 + opts.delta_init, 1.0 - opts.delta_init, size=m)

            init_dir = self.local_search_direction(
                evaluator=evaluator,
                normalizer=normalizer,
                rng=rng,
                x_nominal=x_nominal,
                opts=opts,
                z0=z0,
            )
            state.directions.append(init_dir)
            state.history.append({
                "k": k,
                "f0": float(init_dir.f0),
                "f_best": float(init_dir.f_best),
                "evals_used": int(init_dir.evals_used),
                "v_norm": float(np.linalg.norm(init_dir.v)),
            })

            # Column normalization (keep your original behavior)
            vj = np.asarray(init_dir.v, dtype=float)
            nrm = float(np.linalg.norm(vj))
            if nrm > 0.0:
                V[:, k] = vj / nrm
            else:
                V[:, k] = vj

            # --- Termination check ---
            kk = k + 1
            if kk < min_active:
                continue
            if (kk % int(check_every)) != 0:
                continue

            # SVD of the *current* set of columns only
            Vk = V[:, :kk]
            # full_matrices=False is fine here; we only need left singular vectors & s
            U, s, _ = np.linalg.svd(Vk, full_matrices=False)
            energy = s**2
            total = float(np.sum(energy))

            # Degenerate case: all-zero columns (shouldn't happen, but be safe)
            if total <= 0.0:
                continue

            # Smallest n_keep such that cumulative energy >= threshold
            cum = np.cumsum(energy) / total
            n_keep = int(np.searchsorted(cum, float(energy_threshold)) + 1)

            # Enforce min/max_active on the kept subspace dimension
            n_keep = max(n_keep, min_active)
            if max_active is not None:
                n_keep = min(n_keep, max_active)

            # Track best-so-far (useful if you hit the cap without strict-subset termination)
            U_best, n_keep_best = U, n_keep

            # "Strict subset of existing columns" means n_keep < kk
            # (and also meaningful: you have at least one redundant direction)
            if n_keep < kk:
                # Return an orthonormal basis in R^m (like before).
                # If you want a full m×m basis, request full_matrices=True above,
                # but for downstream you usually only need U[:, :n_keep].
                return state, U, n_keep

        return state, U_best, n_keep_best

    def local_search_direction(self, *, evaluator, normalizer, rng, x_nominal, opts, z0):
        """
        Implements Algorithm 4.1 (Initialization Method for Adaptive Morris Algorithm)
        in normalized coordinates z.

        Center: x^0 := z0
        Sphere radius: delta := opts.delta_init
        Uses ell iterations (each uses one new function evaluation for y; z^+ is model-based).
        """
        z0 = np.asarray(z0, dtype=float)
        delta = float(opts.delta_init)
        if delta <= 0.0:
            raise ValueError("opts.delta_init must be positive for Alg 4.1.")

        # Evaluate f(x^0)
        x0 = normalizer.z_to_x(z0)
        f0 = float(_safe_eval(evaluator, EvalRequest(x=x0, z=z0, stage="init_base", meta={})))
        evals = 1

        # (2) Choose initial x on sphere centered at x^0
        z_x = _sample_on_sphere_about_center(
            rng=rng,
            z0=z0,
            delta=delta
        )
        x_x = normalizer.z_to_x(z_x)
        r = float(_safe_eval(evaluator, EvalRequest(x=x_x, z=z_x, stage="init_x", meta={}))) - f0
        evals += 1

        z_cur = z_x  # this is "x" in the paper's algorithm (point on sphere)

        # (3) Iterate i = 1..ell
        for t in range(int(opts.ell)):
            # (a) Choose y on the same sphere
            z_y = _sample_on_sphere_about_center(
                rng=rng,
                z0=z0,
                delta=delta
            )
            x_y = normalizer.z_to_x(z_y)
            s = float(_safe_eval(
                evaluator,
                EvalRequest(x=x_y, z=z_y, stage="init_y", meta={"iter": int(t)})
            )) - f0
            evals += 1

            dx = z_cur - z0
            dy = z_y - z0

            # (b)(i) p = (1/delta**2) (x-x0)^T (y-x0)
            denom = (delta * delta)
            p = float(np.dot(dx, dy) / denom)
            p = max(-1.0, min(1.0, p))  # numerical safety

            one_minus_p2 = 1.0 - p * p
            if one_minus_p2 <= 1e-12:
                print("colinear")
                # Nearly colinear -> great circle plane is ill-conditioned; skip update
                continue

            # define A,B as in derivation
            A = r - p * s   # (r - p s)
            B = s - p * r   # (s - p r)

            # discriminant: (r-ps)^2 + 2(r-ps)(s-pr)p + (s-pr)^2
            disc = A * A + 2.0 * A * B * p + B * B
            if disc <= 1e-14 * (A*A + B*B + 1.0):   # relative-ish threshold
                continue
            disc = max(0.0, disc)
            sqrt_disc = float(np.sqrt(disc))

            lam0 = sqrt_disc / (2.0 * one_minus_p2)

            def candidate(lam_sign):
                lam = lam_sign * lam0
                scale = 2.0 * lam * one_minus_p2
                a = A / scale
                b = B / scale
                z_plus = z0 + a*dx + b*dy

                # reproject
                u = z_plus - z0
                n = np.linalg.norm(u)
                if n > 0:
                    z_plus = z0 + (delta / n) * u

                r_new = a*r + b*s
                return r_new, z_plus, a, b

            r1, z1, a1, b1 = candidate(+1.0)
            r2, z2, a2, b2 = candidate(-1.0)

            if r2 > r1:
                r_new, z_plus, a, b = r2, z2, a2, b2
            else:
                r_new, z_plus, a, b = r1, z1, a1, b1


            z_cur = z_plus
            r = float(r_new)

        # (4) gradient surrogate column is x - x0
        v = z_cur - z0

        # "best" objective (optional): evaluate at z_cur if you want f_best to be exact.
        # Alg 4.1 itself doesn't require this evaluation; keeping it off preserves eval counts.
        f_best_est = f0 + r

        return InitDirection(
            z0=z0,
            z_best=z_cur,
            v=v,
            f0=float(f0),
            f_best=float(f_best_est),     # model-based estimate consistent with Alg 4.1
            evals_used=int(evals),
            meta={
                "ell": int(opts.ell),
                "delta_init": float(delta),
                "algorithm": "Alg4.1",
                "note": "f_best is f0 + r (model-based); not re-evaluated at z_best",
            },
        )

"""# SVD and Adaptive Step"""

# -----------------------------
# Defaults you can use immediately
# -----------------------------

class SVDBasedSubspaceComputer:
    def compute(self, G, opts):
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        lambdas = s**2
        n_active = _energy_truncation_index(
            lambdas=lambdas,
            tau=opts.energy_threshold,
            min_active=opts.min_active,
            max_active=opts.max_active,
        )
        W1 = U[:, :n_active]
        return SubspaceBasis(W1=W1, lambdas=lambdas, U=U, S=s, Vt=Vt, n_active=n_active)


class EigenWeightedStepSizePolicy:
    def delta(self, lambdas, i, opts):
        lambdas = np.asarray(lambdas, dtype=float) if lambdas is not None else np.array([], dtype=float)
        if lambdas.size == 0:
            return float(0.5 * (opts.delta_min + opts.delta_max))

        l0 = float(lambdas[0])
        lL = float(lambdas[-1])
        li = float(lambdas[i]) if i < lambdas.size else lL

        denom = (l0 - lL)
        w = 0.5 if abs(denom) < 1e-14 else (l0 - li) / denom  # 0 at top, ~1 at tail
        val = float(opts.delta_min + w * (opts.delta_max - opts.delta_min))
        return float(np.clip(val, opts.delta_min, opts.delta_max))

"""# Normalizer"""

# -----------------------------
# Normalizer (±20% nominal -> z in [-1,1]^m)
# -----------------------------

@dataclass(frozen=True)
class PercentBounds:
    """
    Parameter bounds defined as +/- pct around a nominal vector.

    For each i:
      x_low[i]  = x_nom[i] * (1 - pct)
      x_high[i] = x_nom[i] * (1 + pct)

    Notes:
    - If x_nom[i] == 0, the multiplicative rule yields [0,0]. To avoid a degenerate dimension
      you can provide abs_min_width (default 0 -> degenerate allowed).
    - If any dimension is degenerate (x_low == x_high), normalization maps everything to z=0
      for that component and inverse ignores z.
    """
    pct: float = 0.20
    abs_min_width: float = 0.0  # e.g., set to a small positive number if you want non-degenerate bounds for zeros


class NominalPercentNormalizer:
    """
    Affine mapping between physical x and normalized z in [-1,1]^m, using +/-pct nominal bounds.

    z = 2*(x - x_low)/(x_high - x_low) - 1
    x = x_low + 0.5*(x_high - x_low)*(z + 1)

    Degenerate dimensions (width ~ 0) are mapped to z=0.
    """

    def __init__(self, x_nominal, bounds):
        x_nominal = np.asarray(x_nominal, dtype=float)
        if x_nominal.ndim != 1:
            raise ValueError("x_nominal must be a 1D array.")
        if bounds.pct < 0.0:
            raise ValueError("pct must be non-negative.")

        self._x_nom = x_nominal.copy()
        self._pct = float(bounds.pct)
        self._abs_min_width = float(bounds.abs_min_width)

        self._x_low, self._x_high = self._compute_bounds(self._x_nom, self._pct, self._abs_min_width)
        self._width = self._x_high - self._x_low

        # mask for degenerate dims
        self._deg = np.isclose(self._width, 0.0, atol=0.0, rtol=0.0)
        self._safe_width = self._width.copy()
        self._safe_width[self._deg] = 1.0  # avoid division by zero

    @staticmethod
    def _compute_bounds(x_nom, pct, abs_min_width):
        low = x_nom * (1.0 - pct)
        high = x_nom * (1.0 + pct)

        # Ensure low <= high even if x_nom is negative
        x_low = np.minimum(low, high)
        x_high = np.maximum(low, high)

        if abs_min_width > 0.0:
            # If width is too small (e.g., nominal ~ 0), widen symmetrically around nominal
            width = x_high - x_low
            need = width < abs_min_width
            half = 0.5 * abs_min_width
            x_low = np.where(need, x_nom - half, x_low)
            x_high = np.where(need, x_nom + half, x_high)

        return x_low, x_high

    @property
    def m(self):
        return int(self._x_nom.size)

    @property
    def x_low(self):
        return self._x_low.copy()

    @property
    def x_high(self):
        return self._x_high.copy()

    def in_bounds_z(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape != (self.m,):
            raise ValueError(f"z must have shape {(self.m,)}, got {z.shape}")
        return bool(np.all(z >= -1.0) and np.all(z <= 1.0))


    def x_to_z(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.m,):
            raise ValueError(f"x must have shape {(self.m,)}, got {x.shape}")

        # z = 2*(x - x_low)/width - 1
        z = 2.0 * (x - self._x_low) / self._safe_width - 1.0
        z[self._deg] = 0.0
        return z

    def z_to_x(self, z):
        z = np.asarray(z, dtype=float)

        # Single point: (m,)
        if z.shape == (self.m,):
            x = self._x_low + 0.5 * self._width * (z + 1.0)
            x[self._deg] = self._x_nom[self._deg]
            return x

        # Batch: (N, m)
        if z.ndim == 2 and z.shape[1] == self.m:
            x = self._x_low[None, :] + 0.5 * self._width[None, :] * (z + 1.0)
            x[:, self._deg] = self._x_nom[None, self._deg]
            return x

        raise ValueError(f"z must have shape {(self.m,)} or (N,{self.m}), got {z.shape}")



def _normalize_W1_shape(W1, m, n_active=None):
    W1 = np.asarray(W1, dtype=float)

    # If a single vector was provided, treat as one-column basis
    if W1.ndim == 1:
        if W1.size != m:
            raise ValueError(
                f"W1 is 1D with size {W1.size}, but expected size m={m}. "
                "Did you pass the wrong vector?"
            )
        W1 = W1.reshape(m, 1)

    # If it looks transposed (n,m) instead of (m,n), fix it
    if W1.ndim == 2 and W1.shape[0] != m and W1.shape[1] == m:
        W1 = W1.T

    if W1.ndim != 2:
        raise ValueError(f"W1 must be 1D or 2D; got shape {W1.shape}")

    if W1.shape[0] != m:
        raise ValueError(f"W1 has shape {W1.shape}, expected first dimension m={m}.")

    if n_active is not None:
        n_active = int(n_active)
        if n_active < 1:
            raise ValueError("n_active must be >= 1.")
        if n_active > W1.shape[1]:
            raise ValueError(
                f"n_active={n_active} but W1 only has {W1.shape[1]} columns."
            )
        W1 = W1[:, :n_active]

    return W1

SamplingMode = Literal["full_uniform", "active_uniform", "active_gaussian"]


@dataclass(frozen=True)
class SampleBatch:
    z: np.ndarray  # (N, m) normalized parameters in [-1,1]
    x: np.ndarray  # (N, m) physical parameters
    meta: Dict[str, Any]


def _rng(seed=None):
    return np.random.default_rng(seed)

def _uniform(N, d, seed=None):
    """
    Returns points in [-1,1]^d.
    Prefers Sobol if SciPy is available; falls back to iid uniform.
    """
    gen = _rng(seed)
    return gen.uniform(-1.0, 1.0, size=(N, d))


# def _latin_hypercube(N, d, seed=None):
#     """
#     Returns points in [-1,1]^d using LHS if SciPy is available; otherwise uniform.
#     """
#     sampler = qmc.LatinHypercube(d=d, seed=seed)
#     u01 = sampler.random(n=N)
#     return 2.0 * u01 - 1.0


def _orthonormal_complement(W1):
    """
    Given W1 (m, n) with orthonormal columns, return W2 (m, m-n) so [W1 W2] is orthonormal.
    Uses QR on a random matrix projected to complement.
    """
    m, n = W1.shape
    if n == m:
        return np.zeros((m, 0))
    # Start with random and remove components along W1
    G = np.random.default_rng(0).standard_normal((m, m - n))
    G = G - W1 @ (W1.T @ G)
    # Orthonormalize
    W2, _ = np.linalg.qr(G, mode="reduced")
    # Ensure W2 is orthogonal to W1 numerically
    # (optional small re-orth)
    W2 = W2 - W1 @ (W1.T @ W2)
    W2, _ = np.linalg.qr(W2, mode="reduced")
    return W2


def _repair_to_match_y(z, W1, y_target, n_iter=3):
    """
    Simple feasibility repair:
      - adjust z along W1 to bring W1^T z closer to y_target
    This keeps z in [-1,1]^m while roughly preserving desired active coords.
    """
    # Shapes:
    # z: (m,), W1: (m,n), y_target: (n,)
    for _ in range(n_iter):
        y = W1.T @ z
        dy = y_target - y
        # Move along active directions only
        z = z + W1 @ dy
    return z


def sample_training_set(*, normalizer, N, mode, W1=None, n_active=None, inactive_scale=1.0, sampler="uniform",seed=None, repair=True, repair_iters=3):
    """
    Generate training samples.

    Parameters
    ----------
    normalizer
        Must provide:
          - z_to_x(z: (m,) or (N,m)) -> x
        It should map normalized z in [-1,1]^m to physical x.

    N
        Number of samples.

    mode
        - "full_uniform": sample z uniformly in [-1,1]^m
        - "active_uniform": sample y uniformly in [-1,1]^n and inactive η uniformly
        - "active_gaussian": sample y uniformly, inactive η ~ N(0, inactive_scale^2) then clipped

    W1
        Active basis (m, n). Required for active_* modes.
        Columns should be orthonormal.

    n_active
        If provided, use first n_active columns of W1.

    inactive_scale
        Std dev for inactive gaussian, before clipping to [-1,1].

    sampler
        How to sample uniform boxes: "lhs", or "uniform".

    repair
        If True, do a small repair pass so W1^T z matches y better.

    Returns
    -------
    SampleBatch with z (N,m), x (N,m) and metadata.
    """
    gen = _rng(seed)

    m = int(normalizer.m)

    # Choose box sampler for uniform draws
    def draw_box(N_, d_, seed_offset=0):
        s = None if seed is None else (int(seed) + seed_offset)
        return _rng(s).uniform(-1.0, 1.0, size=(N_, d_))

    # Active-subspace-driven sampling
    W1 = _normalize_W1_shape(W1, m=m, n_active=n_active)
    _, n = W1.shape

    if mode == "full_uniform":
        y = draw_box(N, n, seed_offset=11)
        z = (W1 @ y.T).T
        x = normalizer.z_to_x(z)
        return SampleBatch(
            z=z,
            x=x,
            meta={"mode": mode, "sampler": sampler, "seed": seed, "m": m},
        )

    # Build W2
    W2 = _orthonormal_complement(W1)  # (m, m-n)

    # Sample y in [-1,1]^n
    y = draw_box(N, n, seed_offset=21)

    # Sample inactive coords
    if W2.shape[1] == 0:
        eta = np.zeros((N, 0))
    else:
        if mode == "active_uniform":
            eta = draw_box(N, W2.shape[1], seed_offset=31)
        elif mode == "active_gaussian":
            eta = gen.normal(loc=0.0, scale=inactive_scale, size=(N, W2.shape[1]))
            eta = np.clip(eta, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown mode '{mode}'")

    # Assemble z = W1 y + W2 eta
    # Shapes: (m,n)(N,n)^T -> (m,N)^T -> (N,m)
    z = (W1 @ y.T + W2 @ eta.T).T

    if repair:
        z_repaired = np.empty_like(z)
        for i in range(N):
            z_repaired[i] = _repair_to_match_y(
                z=z[i],
                W1=W1,
                y_target=y[i],
                n_iter=repair_iters,
            )
        z = z_repaired

    x = normalizer.z_to_x(z)

    return SampleBatch(
        z=z,
        x=x,
        meta={
            "mode": mode,
            "sampler": sampler,
            "seed": seed,
            "m": m,
            "n_active": n,
            "inactive_dim": int(W2.shape[1]),
            "inactive_scale": inactive_scale,
            "repair": repair,
            "repair_iters": repair_iters,
        },
    )

def create_subspace(x_nom, evaluator):
    rng = np.random.default_rng(67)

    normalizer = NominalPercentNormalizer(x_nominal=x_nom, bounds=PercentBounds(pct=0.2, abs_min_width=0.0))

    # ---- algorithm objects ----
    am = AdaptiveMorris()
    step_policy = EigenWeightedStepSizePolicy()
    subspace = SVDBasedSubspaceComputer()

    ss_opts = SubspaceOptions(
        energy_threshold=0.99,
        min_active=1,
        max_active=10,
    )

    # ---- options ----
    am_opts = AdaptiveMorrisOptions(
        M_max=100,               # columns of G
        delta_min=0.02,         # step sizes in z-space
        delta_max=0.06,
        energy_threshold=0.99,  # truncation per column
        check_every=1,
        basis_tol=0.01,
        patience=3
    )

    # ---- run ----
    state, basis = am.run(
        evaluator=evaluator,
        normalizer=normalizer,
        subspace=subspace,
        x_nominal=x_nom,
        rng=rng,
        opts=am_opts,
        step_policy=step_policy,
        subspace_opts=ss_opts,
        initial_G=None
    )

    return normalizer, state, basis

def sample_subspace(n_samples, normalizer, W1):
    batch = sample_training_set(
        normalizer=normalizer,
        N=n_samples,
        mode="full_uniform",
        W1=W1,
        n_active=1,              # choose dimension n
        sampler="uniform",
        seed=123,
        repair=True,
    )
    return batch