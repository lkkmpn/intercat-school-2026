from typing import overload

import numpy as np
import torch
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints.fix_atoms import FixAtoms
from dscribe.descriptors.soap import SOAP
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances


class GPR:
    def __init__(self, sigma_f: float = 1.0, length_scale: float = 0.5, sigma_n: float = 1e-2):
        """Gaussian process regressor for atomic systems using local
        descriptors, a mean prior, and a squared exponential kernel. Descriptors
        are internally normalized, so variances are with respect to normalized
        descriptors.

        Parameters
        ----------
        sigma_f : float, optional
            Signal variance, by default 1.0.
        length_scale : float, optional
            Length scale, by default 0.5.
        sigma_n : float, optional
            Noise variance, by default 1e-2.
        """

        self.sigma_f = self.initial_sigma_f = sigma_f
        self.length_scale = self.initial_length_scale = length_scale
        self.sigma_n = self.initial_sigma_n = sigma_n

    def fit(
        self,
        descriptors: list[np.ndarray],
        energies: list[float],
        optimize: bool = False,
        optimize_n_restarts: int = 10,
        optimize_bounds_size: float = 1e4,
    ) -> None:
        """Fit the GPR model.

        Parameters
        ----------
        descriptors : list[np.ndarray]
            List of length n_structures, with input descriptors of shape
            (n_atoms, n_features).
        energies : list[float]
            List of length n_structures, with target energies.
        optimize : bool
            Whether to perform hyperparameter optimization, by default False.
        optimize_n_restarts : int
            The number of random initializations of hyperparameters to perform
            during optimization (in addition to the initial values), by default
            10.
        optimize_bounds_size : float
            Factor by which the initial values are scaled down/up to get the
            lower/upper bounds for optimization.
        """

        energies = np.asarray(energies, dtype=float)
        n_atoms_per_structure = np.array([len(d) for d in descriptors])
        self._n_structures = len(descriptors)

        # constant mean prior
        self.e0 = np.mean(energies / n_atoms_per_structure)
        self.y = energies - self.e0 * n_atoms_per_structure

        # stack and normalize descriptors
        X = np.vstack(descriptors)
        self.scale = np.linalg.norm(X, axis=1).mean()
        self.Xhat = X / self.scale  # np.linalg.norm(X, axis=1, keepdims=True)

        # indices of structure starts in stacked descriptors
        self._structure_starts = np.concatenate([[0], np.cumsum(n_atoms_per_structure)[:-1]])

        d2 = euclidean_distances(self.Xhat, self.Xhat, squared=True)

        # optimize hyperparameters
        if optimize:
            theta = self._optimize_hyperparameters(optimize_n_restarts, optimize_bounds_size, d2)
            self.sigma_f, self.length_scale, self.sigma_n = theta

        # solve for alpha
        K_y = self._covariance_matrix(d2, self.sigma_f, self.length_scale, self.sigma_n)  # K + σ_n I
        L = np.linalg.cholesky(K_y)
        self.alpha_per_structure = cho_solve((L, True), self.y)

        self.alpha = np.repeat(self.alpha_per_structure, n_atoms_per_structure)

    @overload
    def predict_local(self, x: np.ndarray, *, return_derivative: bool = False) -> np.ndarray: ...

    @overload
    def predict_local(self, x: np.ndarray, *, return_derivative: bool = True) -> tuple[np.ndarray, np.ndarray]: ...

    def predict_local(
        self, x: np.ndarray, *, return_derivative: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict local energies and optionally energy derivatives. Energy
        derivatives are with respect to non-normalized descriptors.

        Parameters
        ----------
        x : np.ndarray
            Test input descriptor.
        return_derivative : bool, optional
            Whether to return derivatives, by default False.

        Returns
        -------
        np.ndarray | tuple[np.ndarray, np.ndarray]
            Local energies and, if `return_derivative` is True, energy
            derivatives with respect to non-normalized descriptors.
        """

        norm = np.linalg.norm(x, axis=1, keepdims=True)
        xhat = x / self.scale  # norm
        k_star = self._covariance(xhat, self.Xhat)
        e = k_star @ self.alpha

        if not return_derivative:
            return e

        k_weighted = k_star * self.alpha
        de_dxhat = (k_weighted @ self.Xhat - xhat * np.sum(k_weighted, axis=1, keepdims=True)) / self.length_scale**2

        radial_part = np.sum(de_dxhat * xhat, axis=1, keepdims=True)
        de_dx = (de_dxhat - radial_part * xhat) / norm

        return e, de_dx

    def prior(self, n_atoms: int) -> float:
        """Calculate the mean prior for a test input.

        Parameters
        ----------
        n_atoms : int
            Number of atoms of test input.

        Returns
        -------
        float
            Mean prior value.
        """

        return self.e0 * n_atoms

    def predict_energy(self, x: np.ndarray) -> float:
        """Predict the total energy.

        Parameters
        ----------
        x : np.ndarray
            Test input descriptor.

        Returns
        -------
        float
            Total energy prediction.
        """

        e = self.predict_local(x, return_derivative=False)
        return e.sum() + self.e0 * len(x)

    def _sum_over_structures(self, M: np.ndarray) -> np.ndarray:
        """Convert a matrix of shape (n_observations, n_observations) into shape
        (n_structures, n_structures) by summing all entries belonging to each
        pair of structures.
        """

        rows_summed = np.add.reduceat(M, self._structure_starts, axis=0)
        return np.add.reduceat(rows_summed, self._structure_starts, axis=1)

    def _covariance(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Evaluate the covariance function for a squared-exponential kernel."""

        d2 = euclidean_distances(x1, x2, squared=True)  # assumed normalized
        return self.sigma_f**2 * np.exp(-d2 / (2.0 * self.length_scale**2))

    @overload
    def _covariance_matrix(
        self, d2: np.ndarray, sigma_f: float, length_scale: float, sigma_n: float, *, return_derivative: bool = False
    ) -> np.ndarray: ...

    @overload
    def _covariance_matrix(
        self, d2: np.ndarray, sigma_f: float, length_scale: float, sigma_n: float, *, return_derivative: bool = True
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]: ...

    def _covariance_matrix(
        self, d2: np.ndarray, sigma_f: float, length_scale: float, sigma_n: float, *, return_derivative: bool = False
    ) -> np.ndarray | tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate K + σ_n I and optionally its derivative."""

        k = np.exp(-d2 / (2.0 * length_scale**2))
        K = sigma_f**2 * self._sum_over_structures(k)
        noise = sigma_n**2 * np.eye(self._n_structures)

        K_y = K + noise  # K + σ_n I

        if not return_derivative:
            return K_y

        dK_dlog_f = 2.0 * K
        dK_dlog_l = sigma_f**2 * self._sum_over_structures(k * d2) / length_scale**2
        dK_dlog_n = 2.0 * noise

        return K_y, (dK_dlog_f, dK_dlog_l, dK_dlog_n)

    def _log_marginal_likelihood_and_gradient(self, log_theta: np.ndarray, d2: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate the log marginal likelihood and its gradient."""

        sigma_f, length_scale, sigma_n = np.exp(log_theta)

        K_y, dK_y = self._covariance_matrix(d2, sigma_f, length_scale, sigma_n, return_derivative=True)
        L = np.linalg.cholesky(K_y)
        alpha_per_structure = cho_solve((L, True), self.y)

        lml = (
            -0.5 * self.y @ alpha_per_structure
            - np.sum(np.log(np.diag(L)))
            - 0.5 * self._n_structures * np.log(2 * np.pi)
        )

        K_y_inv = cho_solve((L, True), np.eye(self._n_structures))
        W = np.outer(alpha_per_structure, alpha_per_structure) - K_y_inv
        gradient = np.array([0.5 * np.sum(W * dK) for dK in dK_y])

        return lml, gradient

    def _optimize_hyperparameters(self, n_restarts: int, bounds_size: float, d2: np.ndarray) -> np.ndarray:
        """Optimize hyperparameters by maximizing log marginal likelihood.

        Parameters
        ----------
        n_restarts : int
            The number of random initializations of hyperparameters to perform
            during optimization (in addition to the initial values), by default
            10.
        bounds_size : float
            Factor by which the initial values are scaled down/up to get the
            lower/upper bounds for optimization.
        d2 : np.ndarray
            Squared distance matrix between training points.

        Returns
        -------
        np.ndarray
            Optimized hyperparameters `sigma_f`, `length_scale`, `sigma_n`.
        """

        bounds = [
            (np.log(self.initial_sigma_f / bounds_size), np.log(self.initial_sigma_f * bounds_size)),
            (np.log(self.initial_length_scale / bounds_size), np.log(self.initial_length_scale * bounds_size)),
            (np.log(self.initial_sigma_n / bounds_size), np.log(self.initial_sigma_n * bounds_size)),
        ]

        rng = np.random.default_rng(42)
        low, high = np.array(bounds).T
        starts = [np.log([self.initial_sigma_f, self.initial_length_scale, self.initial_sigma_n])]
        starts += [rng.uniform(low, high) for _ in range(10)]

        best = None
        for start in starts:
            res = minimize(self._optimization_objective, start, method='L-BFGS-B', jac=True, bounds=bounds, args=(d2,))
            if best is None or res.fun < best.fun:
                best = res

        return np.exp(best.x)

    def _optimization_objective(self, log_theta: np.ndarray, d2: np.ndarray) -> tuple[float, np.ndarray]:
        try:
            lml, gradient = self._log_marginal_likelihood_and_gradient(log_theta, d2)
            return -lml, -gradient  # maximize log-marginal likelihood -> minimize its negative
        except np.linalg.LinAlgError:
            return 1e10, np.zeros(3)


DTYPE = torch.float64


def to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=DTYPE)


def squared_distances(A: torch.Tensor, B: torch.Tensor):
    d2 = (A**2).sum(dim=1)[:, None] + (B**2).sum(dim=1)[None, :] - 2.0 * A @ B.T
    return d2.clamp_min(0.0)


def farthest_point_sampling(X: torch.Tensor, n_select: int) -> torch.Tensor:
    mean = X.mean(dim=0, keepdim=True)
    first = int(torch.argmin(squared_distances(X, mean)[:, 0]))

    selected = [first]
    min_d2 = squared_distances(X, X[selected[0]][None])[:, 0]
    for _ in range(n_select - 1):
        next_index = int(torch.argmax(min_d2))
        selected.append(next_index)
        min_d2 = torch.minimum(min_d2, squared_distances(X, X[next_index][None])[:, 0])
    return torch.tensor(selected)


class SGPR:
    def __init__(
        self,
        sigma_f: float = 1.0,
        length_scale: float = 0.5,
        sigma_n: float = 1e-2,
        n_inducing: int = 500,
        use_trace_term: bool = True,
        jitter: float = 1e-6,
    ):
        """Sparse Gaussian process regressor for atomic systems using local
        descriptors, a mean prior, and a squared exponential kernel. Descriptors
        are internally normalized, so variances are with respect to normalized
        descriptors.

        Parameters
        ----------
        sigma_f : float, optional
            Signal variance, by default 1.0.
        length_scale : float, optional
            Length scale, by default 0.5.
        sigma_n : float, optional
            Noise variance, by default 1e-2.
        n_inducing : int, optional
            Number of inducing points, by default 500.
        use_trace_term : bool, optional
            Whether to use the trace term, by default True.
        jitter : float, optional
            Jitter for numerical stability, by default 1e-6.
        """

        self.sigma_f = self.initial_sigma_f = sigma_f
        self.length_scale = self.initial_length_scale = length_scale
        self.sigma_n = self.initial_sigma_n = sigma_n

        self.n_inducing = n_inducing
        self.use_trace_term = use_trace_term
        self.jitter = jitter

    def fit(
        self,
        descriptors: list[np.ndarray],
        energies: list[float],
        optimize: bool = False,
        optimize_n_restarts: int = 0,
        optimize_bounds_size: float = 1e4,
    ):
        """Fit the GPR model.

        Parameters
        ----------
        descriptors : list[np.ndarray]
            List of length n_structures, with input descriptors of shape
            (n_atoms, n_features).
        energies : list[float]
            List of length n_structures, with target energies.
        optimize : bool, optional
            Whether to perform hyperparameter optimization, by default False.
        optimize_n_restarts : int
            The number of random initializations of hyperparameters to perform
            during optimization (in addition to the initial values), by default
            10.
        optimize_bounds_size : float
            Factor by which the initial values are scaled down/up to get the
            lower/upper bounds for optimization.
        """

        energies = np.asarray(energies, dtype=float)
        n_atoms_per_structure = np.array([len(d) for d in descriptors])
        self._n_structures = len(descriptors)

        # constant mean prior
        self.e0 = np.mean(energies / n_atoms_per_structure)
        self.y = to_tensor(energies - self.e0 * n_atoms_per_structure)

        # stack and normalize descriptors
        X = to_tensor(np.vstack(descriptors))
        Xhat = X / X.norm(dim=1, keepdim=True)

        # map from descriptors to corresponding structures
        self._owner = torch.repeat_interleave(torch.arange(self._n_structures), torch.as_tensor(n_atoms_per_structure))

        # inducing environments
        n_inducing = min(self.n_inducing, len(Xhat))
        chosen = farthest_point_sampling(Xhat, n_inducing)
        self.inducing_Xhat = Xhat[chosen]

        self._d2_EU = squared_distances(Xhat, self.inducing_Xhat)
        self._d2_UU = squared_distances(self.inducing_Xhat, self.inducing_Xhat)

        if self.use_trace_term:
            starts = np.concatenate(([0], np.cumsum(n_atoms_per_structure)[:-1]))
            self._d2_within = [
                squared_distances(Xhat[s : s + c], Xhat[s : s + c])
                for s, c in zip(starts, n_atoms_per_structure, strict=True)
            ]

        # optimize hyperparameters
        if optimize:
            theta = self._optimize_hyperparameters(optimize_n_restarts, optimize_bounds_size)
            self.sigma_f, self.length_scale, self.sigma_n = theta

        log_theta = torch.log(to_tensor([self.sigma_f, self.length_scale, self.sigma_n]))

        # solve for alpha
        with torch.no_grad():
            lml, L_U, L_B, A = self._log_marginal_likelihood(log_theta)
            A_y = (A @ self.y)[:, None]
            self.inducing_weights = torch.linalg.solve_triangular(
                L_U.T,
                torch.cholesky_solve(A_y, L_B),
                upper=True,
            )[:, 0] / (self.sigma_n**2)

        self.log_marginal_likelihood = lml.item()

        del self._d2_EU, self._d2_UU
        if self.use_trace_term:
            del self._d2_within

    @overload
    def predict_local(self, x: np.ndarray, *, return_derivative: bool = False) -> np.ndarray: ...

    @overload
    def predict_local(self, x: np.ndarray, *, return_derivative: bool = True) -> tuple[np.ndarray, np.ndarray]: ...

    def predict_local(self, x: np.ndarray, *, return_derivative: bool):
        """Predict local energies and optionally energy derivatives. Energy
        derivatives are with respect to non-normalized descriptors.

        Parameters
        ----------
        x : np.ndarray
            Test input descriptor.
        return_derivative : bool
            Whether to return derivatives, by default False.

        Returns
        -------
        np.ndarray | tuple[np.ndarray, np.ndarray]
            Local energies and, if `return_derivative` is True, energy
            derivatives with respect to non-normalized descriptors.
        """

        if not return_derivative:
            with torch.no_grad():
                return self._local_energies(to_tensor(x)).numpy()

        else:
            x = to_tensor(x).requires_grad_(True)
            e = self._local_energies(x)
            (f,) = torch.autograd.grad(e.sum(), x)
            return e.detach().numpy(), f.numpy()

    def prior(self, n_atoms: int) -> float:
        """Calculate the mean prior for a test input.

        Parameters
        ----------
        n_atoms : int
            Number of atoms of test input.

        Returns
        -------
        float
            Mean prior value.
        """

        return self.e0 * n_atoms

    def predict_energy(self, x: np.ndarray):
        """Predict the total energy.

        Parameters
        ----------
        x : np.ndarray
            Test input descriptor.

        Returns
        -------
        float
            Total energy prediction.
        """

        e = self.predict_local(x, return_derivative=False)
        return e.sum() + self.e0 * len(x)

    def _local_energies(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate local energies.

        Parameters
        ----------
        x : torch.Tensor
            Test input descriptor.

        Returns
        -------
        torch.Tensor
            Local energies.
        """

        xhat = x / x.norm(dim=1, keepdim=True)
        d2 = squared_distances(xhat, self.inducing_Xhat)
        k_star = self.sigma_f**2 * torch.exp(-d2 / (2 * self.length_scale**2))
        e = k_star @ self.inducing_weights
        return e

    def _log_marginal_likelihood(self, log_theta: torch.Tensor):
        """Evaluate the log marginal likelihood and related intermediate terms."""

        sigma_f, length_scale, sigma_n = torch.exp(log_theta)
        n, M = self._n_structures, len(self.inducing_Xhat)

        K_UU = sigma_f**2 * torch.exp(-self._d2_UU / (2 * length_scale**2))
        K_UU = K_UU + self.jitter * sigma_f**2 * torch.eye(M, dtype=DTYPE)
        k_EU = sigma_f**2 * torch.exp(-self._d2_EU / (2 * length_scale**2))
        K_SU = torch.zeros(n, M, dtype=DTYPE).index_add(0, self._owner, k_EU)

        L_U = torch.linalg.cholesky(K_UU)
        A = torch.linalg.solve_triangular(L_U, K_SU.T, upper=False)

        B = torch.eye(M, dtype=DTYPE) + A @ A.T / (sigma_n**2)
        L_B = torch.linalg.cholesky(B)
        c = torch.linalg.solve_triangular(L_B, (A @ self.y)[:, None], upper=False)[:, 0]
        quadratic = self.y @ self.y / (sigma_n**2) - c @ c / (sigma_n**4)
        log_det = n * torch.log(sigma_n**2) + 2.0 * torch.log(torch.diagonal(L_B)).sum()
        lml = -0.5 * (quadratic + log_det + n * np.log(2 * np.pi))

        if self.use_trace_term:
            trace_K = sum((sigma_f**2 * torch.exp(-d2 / (2 * length_scale**2))).sum() for d2 in self._d2_within)
            lml = lml - (trace_K - (A**2).sum()) / (2 * (sigma_n**2))

        return lml, L_U, L_B, A

    def _optimize_hyperparameters(self, n_restarts: int, bounds_size: float) -> np.ndarray:
        """Optimize hyperparameters by maximizing log marginal likelihood.

        Parameters
        ----------
        n_restarts : int
            The number of random initializations of hyperparameters to perform
            during optimization (in addition to the initial values), by default
            10.
        bounds_size : float
            Factor by which the initial values are scaled down/up to get the
            lower/upper bounds for optimization.

        Returns
        -------
        np.ndarray
            Optimized hyperparameters `sigma_f`, `length_scale`, `sigma_n`.
        """

        bounds = [
            (np.log(self.initial_sigma_f / bounds_size), np.log(self.initial_sigma_f * bounds_size)),
            (np.log(self.initial_length_scale / bounds_size), np.log(self.initial_length_scale * bounds_size)),
            (np.log(self.initial_sigma_n / bounds_size), np.log(self.initial_sigma_n * bounds_size)),
        ]

        rng = np.random.default_rng(42)
        low, high = np.array(bounds).T
        starts = [np.log([self.initial_sigma_f, self.initial_length_scale, self.initial_sigma_n])]
        starts += [rng.uniform(low, high) for _ in range(n_restarts)]

        best = None
        for start in starts:
            res = minimize(self._optimization_objective, start, method='L-BFGS-B', jac=True, bounds=bounds)
            print(res)
            if best is None or res.fun < best.fun:
                best = res

        return np.exp(best.x)

    def _optimization_objective(self, log_theta: np.ndarray) -> tuple[float, np.ndarray]:
        log_theta = torch.tensor(log_theta, dtype=DTYPE, requires_grad=True)
        try:
            lml, *_ = self._log_marginal_likelihood(log_theta)
            lml.backward()
        except RuntimeError:
            return 1e10, np.zeros(3)

        if not torch.isfinite(lml):
            return 1e10, np.zeros(3)

        return -lml.item(), -log_theta.grad.numpy()


class SOAPGPRCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(self, soap: SOAP, gpr: GPR | SGPR, *, exclude_fixed: bool = True):
        """ASE calculator wrapper for a GPR using a SOAP descriptor as input.

        Parameters
        ----------
        soap : SOAP
            SOAP descriptor definition.
        gpr : GPR | SGPR
            Trained GPR model.
        exclude_fixed : bool, optional
            Whether to exclude fixed atoms from SOAP derivative calculation,
            speeding up the calculation. By default True.
        """

        super().__init__()

        self.soap = soap
        self.gpr = gpr
        self.exclude_fixed = exclude_fixed

    def calculate(self, atoms: Atoms | None = None, properties=('energy',), system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)

        if self.exclude_fixed:
            unfixed_indices = set(range(len(self.atoms)))

            for constraint in atoms.constraints:
                if isinstance(constraint, FixAtoms):
                    unfixed_indices -= set(constraint.get_indices())

            unfixed_indices = sorted(unfixed_indices)

            x = self.soap.create(self.atoms)
            dx = np.zeros((len(self.atoms), len(self.atoms), 3, self.soap.get_number_of_features()))
            dx[:, unfixed_indices, :, :] = self.soap.derivatives_single(
                self.atoms, centers=None, indices=unfixed_indices, attach=True, return_descriptor=False
            )

        else:
            dx, x = self.soap.derivatives_single(self.atoms, attach=True, return_descriptor=True)

        e, de_dx = self.gpr.predict_local(x, return_derivative=True)

        self.results.update(
            {
                'energy': np.sum(e) + self.gpr.prior(len(self.atoms)),
                'forces': -np.einsum('if,ikcf->kc', de_dx, dx),
            }
        )
