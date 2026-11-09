"""Solution template."""

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel, DotProduct
import matplotlib.pyplot as plt
from scipy.stats import norm


SEARCH_BOUNDS = np.array([[0.0, 10.0]])
MAX_SAFE_SA = 4.0
PLOT_DIAGNOSTICS = False  # Debug only: plot the first trial, then stop.


class SafeBOAgent:
    """Choose safe queries and return a final point using noisy observations."""

    def __init__(self):
        self.explored_points = np.zeros((0,3))
        self.recommended_points = np.zeros(0)

        self.confidence_width_scale = 3.0
        self.safety_confidence_scale = 3.0
        self.safety_margin = 0.001
        self.constraint_prior_mean = MAX_SAFE_SA
        self.min_points_for_kernel_fit = 4
        self.candidate_grid = np.linspace(*SEARCH_BOUNDS[0], 2001)

        self.noise_f = 0.15**2
        self.noise_v = 0.0001**2

        self.matern_nu_f = 2.5
        self.matern_nu_v = np.inf
        self.matern_length_f = 1
        self.matern_length_v = 0.5

        self.delta = 0.05

        # These are starting bounds to validate, not task-provided guarantees.
        self.kernel_f = ConstantKernel(0.5, (0.05, 5.0)) * Matern(
            nu=self.matern_nu_f,
            length_scale=self.matern_length_f,
            length_scale_bounds=(0.1, 20.0),
        )
        self.kernel_v = DotProduct(sigma_0_bounds="fixed") + ConstantKernel(
            np.sqrt(2.0), (0.1, 10.0)
        ) * Matern(
            nu=self.matern_nu_v,
            length_scale=self.matern_length_v,
            length_scale_bounds=(0.1, 20.0),
        )

        self.gpr_estim_f = GaussianProcessRegressor(
            kernel=self.kernel_f, alpha=self.noise_f,
            optimizer=None, n_restarts_optimizer=3, random_state=0,
        )
        self.gpr_estim_v = GaussianProcessRegressor(
            kernel=self.kernel_v, alpha=self.noise_v,
            optimizer=None, n_restarts_optimizer=3, random_state=1,
        )

        self.step_count = 0
                

    def _fit_models(self) -> None:
        """Keep the prior kernels until enough distinct inputs are available."""
        explored_x = self.explored_points[:, :1]
        fit_kernels = np.unique(explored_x).size >= self.min_points_for_kernel_fit
        for model in (self.gpr_estim_f, self.gpr_estim_v):
            model.optimizer = "fmin_l_bfgs_b" if fit_kernels else None
            if fit_kernels and hasattr(model, "kernel_"):
                model.kernel = model.kernel_
        self.gpr_estim_f.fit(explored_x, self.explored_points[:, 1])
        self.gpr_estim_v.fit(
            explored_x, self.explored_points[:, 2] - self.constraint_prior_mean
        )

    def _predict_safety(self, candidates: np.ndarray):
        """Return constraint predictions on the original SA scale."""
        v_mean, v_std = self.gpr_estim_v.predict(candidates, return_std=True)
        return v_mean + self.constraint_prior_mean, v_std

    def suggest_next(self) -> np.ndarray:
        """Return the next query as an array of shape (1, 1)."""
        if self.explored_points.size == 0:
            raise RuntimeError("Observe the supplied initial safe point before querying.")
        self._fit_models()
        next_rec = self.optimize_acquisition()
        self.recommended_points = np.append(self.recommended_points, next_rec)
        if PLOT_DIAGNOSTICS:
            self.plot_diagnostics(include_recommendation=True)
        return next_rec

    def optimize_acquisition(self, restarts: int = 20) -> np.ndarray:
        """Search a grid plus observations inside the seed's estimated safe region.

        ``restarts`` is retained for compatibility; no random restarts are used.
        """
        candidates = np.unique(np.concatenate((
            self.candidate_grid, self.explored_points[:, 0]
        )))[:, None]
        scores = self.acquisition_function(candidates)

        # The first observation is the task-supplied, known-safe seed. It is
        # available even if the GP cannot currently certify any other point.
        seed_x = self.explored_points[0, 0]
        seed_idx = int(np.searchsorted(candidates[:, 0], seed_x))
        seed_mean, seed_std = self.gpr_estim_f.predict(
            candidates[seed_idx:seed_idx + 1], return_std=True
        )
        scores[seed_idx] = (
            seed_mean.item() + self.confidence_width_scale * seed_std.item()
        )

        # In one dimension, the contiguous safe grid component approximates
        # the safely reachable region. Do not jump across uncertified gaps.
        safe = np.isfinite(scores)
        left = right = seed_idx
        while left > 0 and safe[left - 1]:
            left -= 1
        while right + 1 < len(candidates) and safe[right + 1]:
            right += 1
        scores[:left] = -np.inf
        scores[right + 1:] = -np.inf
        best_idx = int(np.argmax(scores))
        return candidates[best_idx:best_idx + 1].copy()

    def acquisition_function(self, candidates: np.ndarray) -> np.ndarray:
        """Objective UCB, with uncertified candidates excluded."""
        f_mean, f_std = self.gpr_estim_f.predict(candidates, return_std=True)
        v_mean, v_std = self._predict_safety(candidates)
        v_ucb = v_mean + self.safety_confidence_scale * v_std
        f_ucb = f_mean + self.confidence_width_scale * f_std
        safe = v_ucb < MAX_SAFE_SA - self.safety_margin
        return np.where(safe, f_ucb, -np.inf)

    def observe(
        self, point: np.ndarray, objective_value: float, safety_value: float
    ) -> None:
        """Record a query and its noisy logP and SA observations."""
        
        # The checker may pass observations as single-element arrays.
        new_point = np.array([[
            np.asarray(point, dtype=float).item(),
            np.asarray(objective_value, dtype=float).item(),
            np.asarray(safety_value, dtype=float).item(),
        ]])
        if self.explored_points.size == 0:
            self.explored_points = new_point
        else:
            self.explored_points = np.vstack([self.explored_points, new_point])


    def best_safe_point(self) -> np.ndarray:
        """Return your predicted safe optimum as an array of shape (1, 1)."""
        safe_mask = self.explored_points[:, 2] < MAX_SAFE_SA
        
        if np.any(safe_mask):
            safe_points = self.explored_points[safe_mask]
            best_idx = np.argmax(safe_points[:, 1])
            return safe_points[best_idx:best_idx + 1, :1]
        else:
            safest_idx = np.argmin(self.explored_points[:, 2])
            return self.explored_points[safest_idx:safest_idx + 1, :1]

    def plot_diagnostics(self, include_recommendation: bool = True) -> None:
        """Optional plot; called by checker when PLOT_DIAGNOSTICS is True.

        Save plots that you create here under the docker results directory (default is "/results"). Example:
        ```
        # plot stuff
        plt.savefig(f"/results/{filename}.pdf")
        ```
        """
        fig = plt.figure(figsize=(20, 8), dpi=80)
        (ax1, ax2, ax3) = fig.subplots(1,3)

        x_grid = np.linspace(0,10,200).reshape(-1,1)
        f_mean, f_std = self.gpr_estim_f.predict(x_grid, return_std=True)
        v_mean, v_std = self._predict_safety(x_grid)
        acquisition_func = self.acquisition_function(x_grid)

        ax1.scatter(self.explored_points[:,0], self.explored_points[:,1], c='darkorange', s=100, zorder=5, label='Exploration data')
        ax1.plot(x_grid, f_mean, 'b-', linewidth=2, label='Mean prediction for f')
        ax1.fill_between(x_grid.ravel(), 
            f_mean - self.confidence_width_scale*f_std, 
            f_mean + self.confidence_width_scale*f_std, 
            alpha=0.3, 
            color='navajowhite',
            label='confidence interval')
        ax1.set_title("Data Exploration of f")
        ax1.set_ylim(-10,10)

        ax2.scatter(self.explored_points[:,0], self.explored_points[:,2], c='darkorange', s=100, zorder=5, label='Exploration data')
        ax2.plot(x_grid, v_mean, 'b-', linewidth=2, label='Mean prediction for v')
        ax2.fill_between(x_grid.ravel(), 
            v_mean - self.safety_confidence_scale*v_std, 
            v_mean + self.safety_confidence_scale*v_std, 
            alpha=0.3, 
            color='navajowhite',
            label='confidence interval')
        ax2.set_title("Data Exploration of v")
        ax2.hlines(4, 0, 10)
        ax2.set_ylim(-10,10)
        
        ax3.plot(x_grid, acquisition_func, 'b-', linewidth=2, label='Mean prediction for v')
        ax3.set_title("Constrained Acquisition function")
        # ax3.set_ylim(0,3)

        if include_recommendation:
            last_rec = self.recommended_points[-1].reshape(1,-1)
            rec_mean_f = self.gpr_estim_f.predict(last_rec)
            rec_mean_v, _ = self._predict_safety(last_rec)

            ax1.scatter(last_rec, rec_mean_f, c='red', s=100, zorder=5, label='Recommended Point')
            ax2.scatter(last_rec, rec_mean_v, c='red', s=100, zorder=5, label='Recommended Point')
    
        # plt.show()
        fig.savefig(f"./images/recommendation_graph_{self.step_count}.png", bbox_inches="tight")
        self.step_count+=1


# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def in_search_bounds(point: np.ndarray) -> bool:
    """Check whether a point is inside the search interval."""
    point = np.atleast_2d(point)
    return np.all(point >= SEARCH_BOUNDS[:, 0]) and np.all(point <= SEARCH_BOUNDS[:, 1])


def toy_objective(point: np.ndarray) -> float:
    """Toy logP objective."""
    midpoint = SEARCH_BOUNDS[:, 0] + 0.5 * (SEARCH_BOUNDS[:, 1] - SEARCH_BOUNDS[:, 0])
    return -float(np.linalg.norm(point - midpoint))


def toy_safety(point: np.ndarray) -> float:
    """Toy SA value."""
    return 2.0


def initial_safe_point() -> np.ndarray:
    """Choose an initial point with toy SA below the threshold."""
    grid = np.linspace(*SEARCH_BOUNDS[0], 4000)[:, None]
    safe_grid = grid[np.array([toy_safety(point) < MAX_SAFE_SA for point in grid])]
    np.random.seed(0)
    np.random.shuffle(safe_grid)
    return safe_grid[:1]


def main() -> None:
    """Illustration only; the checker does not call this function."""
    agent = SafeBOAgent()
    point = initial_safe_point()
    agent.observe(point, toy_objective(point), toy_safety(point))

    for _ in range(20):
        point = agent.suggest_next()
        assert point.shape == (1, SEARCH_BOUNDS.shape[0]), (
            f"suggest_next() must return shape (1, {SEARCH_BOUNDS.shape[0]})"
        )
        agent.observe(
            point,
            toy_objective(point) + np.random.randn() * 0.1,
            toy_safety(point) + np.random.randn() * 0.01,
        )

    solution = agent.best_safe_point()
    assert in_search_bounds(solution), "best_safe_point() returned a point outside the domain"
    print(f"Proposed solution: {solution}; toy objective: {toy_objective(solution)}")


if __name__ == "__main__":
    main()
