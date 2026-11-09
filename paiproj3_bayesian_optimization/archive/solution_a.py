"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel, DotProduct
import matplotlib.pyplot as plt
from scipy.stats import norm
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        self.explored_points = np.zeros((0,3))
        self.recommended_points = np.zeros(0)
        self.safe_points = {}
        self.candidate_decisions = {}
        self.potential_maximizers = {}

        self.constraint_penalty = 0.25
        self.confidence_width_scale = 2.0

        self.noise_f = np.sqrt(0.15)
        self.noise_v = np.sqrt(0.0001)

        self.matern_nu_f = 2.5
        self.matern_nu_v = 2.5
        self.matern_length_f = .5
        self.matern_length_v = 10
        self.lipschitz_const_f = np.sqrt(np.divide(2*self.matern_nu_f, self.matern_nu_f-1))/self.matern_length_f

        self.kernel_f = ConstantKernel(np.sqrt(0.5))      * Matern(nu=self.matern_nu, length_scale_bounds=(1e-10,1e10), length_scale=.5) + WhiteKernel(noise_level=self.noise_f, noise_level_bounds=(1e-10,1e5))
        self.kernel_v = DotProduct() + ConstantKernel(np.power(2, 0.25)) * Matern(nu=self.matern_nu, length_scale_bounds=(1e-10,1e10), length_scale=10) + WhiteKernel(noise_level=self.noise_v, noise_level_bounds=(1e-10,1e5))

        self.gpr_estim_f = GaussianProcessRegressor(n_restarts_optimizer=10,kernel=self.kernel_f)
        self.gpr_estim_v = GaussianProcessRegressor(n_restarts_optimizer=10,kernel=self.kernel_v)

        self.step_count = 0


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: np.ndarray
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        for safe_point in self.safe_points[f'step_{self.step_count}']:
            self.safe_points[f'step_{self.step_count+1}'] = np.vstack([self.explored_points])

        explored_x = self.explored_points[:,0].reshape(-1,1)
        explored_f = self.explored_points[:,1].reshape(-1,1)
        explored_v = self.explored_points[:,2].reshape(-1,1)

        self.gpr_estim_f.fit(explored_x, explored_f)
        self.gpr_estim_v.fit(explored_x, explored_v)

        # next_rec = np.clip(self.optimize_acquisition_function() + self.noise_v*np.random.randn(), DOMAIN[:, 0], DOMAIN[:, 1])
        next_rec = self.optimize_acquisition_function()

        self.recommended_points = np.append(self.recommended_points, next_rec)

        self.plot(plot_recommendation=True)
        self.step_count+=1

        return self.recommended_points[-1].reshape(1,1)

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick the best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        # Acquisition function for GP-UCB
        # f_mean, f_std = self.gpr_estim_f.predict(x, return_std=True)
        # objective = f_mean + np.sqrt(self.confidence_width_scale)*f_std

        # Acquisition function for SafeOpt
        objective = 0

        return objective

    def add_data_point(self, x: float, f: float, v: float):
        new_point = np.array([[float(x), f, v]])
        if self.explored_points.size == 0:
            self.explored_points = new_point
            self.safe_points[f"step_{self.step_count}"] = new_point
        else:
            self.explored_points = np.vstack([self.explored_points, new_point])

    def get_solution(self):
        # Return the point with highest f that satisfies safety constraint
        safe_mask = self.explored_points[:, 2] < SAFETY_THRESHOLD
        if np.any(safe_mask):
            safe_points = self.explored_points[safe_mask]
            best_idx = np.argmax(safe_points[:, 1])
            return safe_points[best_idx, 0]
        else:
            # Fallback: return safest point
            safest_idx = np.argmin(self.explored_points[:, 2])
            return self.explored_points[safest_idx, 0]

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        fig = plt.figure(figsize=(20, 8), dpi=80)
        (ax1, ax2, ax3) = fig.subplots(1,3)

        x_grid = np.linspace(0,10,200).reshape(-1,1)
        f_mean, f_std = self.gpr_estim_f.predict(x_grid, return_std=True)
        v_mean, v_std = self.gpr_estim_v.predict(x_grid, return_std=True)
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
            v_mean - self.confidence_width_scale*v_std, 
            v_mean + self.confidence_width_scale*v_std, 
            alpha=0.3, 
            color='navajowhite',
            label='confidence interval')
        ax2.set_title("Data Exploration of v")
        ax2.set_ylim(-10,10)
        
        ax3.plot(x_grid, acquisition_func, 'b-', linewidth=2, label='Mean prediction for v')
        ax3.set_title("Constrained Acquisition function")
        ax3.set_ylim(-10,10)

        if plot_recommendation:
            last_rec = self.recommended_points[-1].reshape(1,-1)
            rec_mean_f = self.gpr_estim_f.predict(last_rec)
            rec_mean_v = self.gpr_estim_v.predict(last_rec)

            ax1.scatter(last_rec, rec_mean_f, c='red', s=100, zorder=5, label='Recommended Point')
            ax2.scatter(last_rec, rec_mean_v, c='red', s=100, zorder=5, label='Recommended Point')
    
        # plt.show()
        fig.savefig(f"./images/recommendation_graph_{self.step_count}.png", bbox_inches="tight")



# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()


        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()

        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
