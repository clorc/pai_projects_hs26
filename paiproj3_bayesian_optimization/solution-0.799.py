"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, Matern, ConstantKernel as C
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
# Bayesian optimisation algorithm
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # We store the observed data points in lists
        self.X_data = []
        self.F_data = []  # Objective (logP) values
        self.V_data = []  # Constraint (SA) values

        # --- GP Model Definitions ---
        # We use variance (std_dev^2) for the 'alpha' parameter
        self.f_noise_var = 0.15**2
        self.v_noise_var = 0.0001**2

         # --- Prior mean ---
        # We'll implement a constant prior mean by centering targets when fitting
        # and adding it back when predicting.
        self.prior_mean = 4.0

        # Define kernels based on PDF recommendation (Matérn) 
        # We allow the GPR to fit the variance and length_scale by setting bounds
        # kernel_f = C(1.0, (1e-3, 1e3)) * Matern(
        #     length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5
        # )
        # kernel_v = WhiteKernel()+C(1.0, (1e-3, 1e3)) * Matern(
        #     length_scale=0.5, length_scale_bounds=(1e-2, 1e2), nu=2.5
        # )

        ## Matern length scale 10: 0.6574409349300316
        ## Matern length scale 0.5: 0.62236774879724
        ## Matern length scale 1: 0.6177421269529052

        ## RGB length scale 0.5: 0.7192093740970291 , beta=1.96
        ## RGB length scale 0.5: 0.7626642278083718, beta = 2.0


        kernel_f = C(0.5, (1e-3, 1e3)) * RBF(
            length_scale=0.5, length_scale_bounds=(1e-2, 1e2)
        )
        kernel_v = C(math.sqrt(2), (1e-3, 1e3)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
        )

        # 1. GP for the objective function f (logP)
        self.gp_f = GaussianProcessRegressor(
            kernel=kernel_f,
            alpha=self.f_noise_var,
            n_restarts_optimizer=15  # Run optimizer multiple times to find good kernel params
        )

        # 2. GP for the constraint function v (SA)
        self.gp_v = GaussianProcessRegressor(
            kernel=kernel_v,
            alpha=self.v_noise_var,
            n_restarts_optimizer=15,
            normalize_y=False
        )

        # --- Algorithm State ---
        self.is_fitted = False  # Flag to track if GPs have been fitted
        self.beta = 2.1     # Parameter for UCB (balances exploration/exploitation)
        self.lambda_penalty = 90.0 # next 100, beta 2.0

        self.recommended_points = np.zeros(0)
        self.step_count = 0

        pass

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

        # We assume add_data_point has been called at least once (as in main())
        # We simply optimize the acquisition function to find the next point.
        x_opt = self.optimize_acquisition_function()

        self.recommended_points = np.append(self.recommended_points, x_opt)
        self.plot(plot_recommendation=True)
        
        # Return as a 2D array, shape (1, 1) as required by the checker
        return np.array([[x_opt]])

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            # We minimize the negative acquisition function
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
        # --- 1. Get Posterior for Objective f (logP) ---
        mu_f, std_f = self.gp_f.predict(x, return_std=True)
        # Get scalar values (optimizer expects a scalar return)
        mu_f_scalar = mu_f[0]
        std_f_scalar = std_f[0]
        
        # --- 2. Get Posterior for Constraint v (SA) ---
        # mu_v, std_v = self.gp_v.predict(x, return_std=True)
        mu_v_centered, std_v = self.gp_v.predict(x, return_std=True)
        mu_v = mu_v_centered + self.prior_mean
        mu_v_scalar = mu_v[0]
        std_v_scalar = std_v[0]

        # # --- 3. Calculate Acquisition Components ---

        # # Component A: Upper Confidence Bound (UCB) for objective
        # # This balances exploiting known high values (mu_f) and
        # # exploring uncertain regions (std_f).
        # ucb = mu_f_scalar + self.beta * std_f_scalar

        # #  Component B: Probabilityof Safety (PoS)
        # # P(v(x) <= SAFETY_THRESHOLD)
        # # We use the CDF of the normal distribution from our constraint GP.
        
        # # Add small jitter to std_v to avoid division by zero
        # std_v_safe = np.maximum(std_v_scalar, 1e-9)
        # # Standardize the variable
        # z = (SAFETY_THRESHOLD - mu_v_scalar) / std_v_safe
        # # Calculate probability using the Cumulative Distribution Function (CDF)
        # pos = norm.cdf(z)
        
        # # --- 4. Final Acquisition Function ---
        # # We multiply the UCB by the probability of being safe.
        # # This ensures we only explore/exploit in regions that are
        # # likely to be safe. 
        # af_value = ucb * pos *0.9

        ## --- Lagrangian relaxation ---
        # UCB for objective (we want to maximize this)
        ucb_f = mu_f + self.beta * std_f

        # For constraint, we want to be conservative (pessimistic)
        # Use UCB to get upper bound on constraint (worst case)
        ucb_v = mu_v + self.beta * std_v
        
        # Lagrangian relaxation acquisition function
        # max f(x) - λ * max(v(x) - κ, 0)
        constraint_violation = np.maximum(ucb_v - SAFETY_THRESHOLD, 0.0)
        af_value = ucb_f - self.lambda_penalty * constraint_violation
        
        
        return af_value # Return scalar float

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float or np.ndarray
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        # Store the new data point
        self.X_data.append(float(x))
        self.F_data.append(float(f))
        self.V_data.append(float(v))

        # Prepare data for GP fitting
        # X must be 2D (n_samples, n_features=1)
        # F and V must be 1D (n_samples,)
        X = np.array(self.X_data).reshape(-1, 1)
        F = np.array(self.F_data)
        V_centered = np.array(self.V_data) - self.prior_mean

        # Re-fit both GP models with all available data
        self.gp_f.fit(X, F)
        self.gp_v.fit(X, V_centered)

        self.is_fitted = True # Mark that we have a fitted model

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        # Get all observed X points
        X_obs = np.array(self.X_data).reshape(-1, 1)
        
        # Predict the *mean* objective and constraint values at all observed points
        mu_f_all = self.gp_f.predict(X_obs)
        mu_v_centered = self.gp_v.predict(X_obs)
        mu_v_all = mu_v_centered + self.prior_mean
        #mu_v_all = self.gp_v.predict(X_obs)
        
        # Find the indices of all points that are predicted to be safe
        safe_indices = np.where(mu_v_all <= SAFETY_THRESHOLD)[0]
        
        if len(safe_indices) == 0:
            # This should not happen given the initial safe point.
            # As a robust fallback, return the point with the lowest predicted constraint.
            best_index = np.argmin(mu_v_all)
            return self.X_data[best_index]
        
        # From the set of safe points, find the one with the highest
        # predicted objective value (mu_f).
        safe_objectives = mu_f_all[safe_indices]
        best_safe_point_index = np.argmax(safe_objectives)
        
        # Get the original index of this best safe point
        best_original_index = safe_indices[best_safe_point_index]
        
        # Return the corresponding x value
        return self.X_data[best_original_index]

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
        f_mean, f_std = self.gp_f.predict(x_grid, return_std=True)
        v_mean, v_std = self.gp_v.predict(x_grid, return_std=True)
        acquisition_func = self.acquisition_function(x_grid)

        ax1.scatter(self.X_data, self.F_data, c='darkorange', s=100, zorder=5, label='Exploration data')
        ax1.plot(x_grid, f_mean, 'b-', linewidth=2, label='Mean prediction for f')
        ax1.fill_between(x_grid.ravel(), 
            f_mean - self.beta*f_std, 
            f_mean + self.beta*f_std, 
            alpha=0.3, 
            color='navajowhite',
            label='confidence interval')
        ax1.set_title("Data Exploration of f")
        ax1.set_ylim(-10,10)

        ax2.scatter(self.X_data, self.V_data, c='darkorange', s=100, zorder=5, label='Exploration data')
        ax2.plot(x_grid, v_mean, 'b-', linewidth=2, label='Mean prediction for v')
        ax2.fill_between(x_grid.ravel(), 
            v_mean - self.beta*v_std, 
            v_mean + self.beta*v_std, 
            alpha=0.3, 
            color='navajowhite',
            label='confidence interval')
        ax2.set_title("Data Exploration of v")
        ax2.hlines(4, 0, 10)
        ax2.set_ylim(-10,10)
        
        ax3.plot(x_grid, acquisition_func, 'b-', linewidth=2, label='Mean prediction for v')
        ax3.set_title("Constrained Acquisition function")
        # ax3.set_ylim(0,3)

        if plot_recommendation:
            last_rec = self.recommended_points[-1].reshape(1,-1)
            rec_mean_f = self.gp_f.predict(last_rec)
            rec_mean_v = self.gp_v.predict(last_rec)

            ax1.scatter(last_rec, rec_mean_f, c='red', s=100, zorder=5, label='Recommended Point')
            ax2.scatter(last_rec, rec_mean_v, c='red', s=100, zorder=5, label='Recommended Point')
    
        # plt.show()
        fig.savefig(f"./images/recommendation_graph_{self.step_count}.png", bbox_inches="tight")
        self.step_count+=1




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
