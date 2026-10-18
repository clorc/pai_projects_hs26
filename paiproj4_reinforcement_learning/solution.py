import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.utils import seeding
from utils import ReplayBuffer, get_env, run_episode


class MLP(nn.Module):
    '''
    A simple ReLU MLP constructed from a list of layer widths.
    '''
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i, (in_size, out_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(in_size, out_size))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Critic(nn.Module):
    '''
    Simple MLP Q-function.
        '''
    def __init__(self, obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        self.net = MLP([obs_size + action_size] + ([num_units] * num_layers) + [1])

        #####################################################################

    def forward(self, x, a):
        #####################################################################
        # TODO: code the forward pass
        # the critic receives a batch of observations and a batch of actions
        # of shape (batch_size x obs_size) and batch_size x action_size) respectively
        # and output a batch of values of shape (batch_size x 1)
        
        value = torch.zeros(x.shape[0], 1, device=x.device)

        #####################################################################
        return value

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    '''
    Gaussian stochastic actor.
    Outputs a (mean, std) of a normal dist. 
    '''
    def __init__(self, action_low, action_high,  obs_size, action_size, num_layers, num_units):
        super().__init__()
        #####################################################################
        # TODO: define the network layers
        # The network should output TWO values for each action dimension:
        # The mean (mu) and the log standard deviation (log_std)
        # So, the final layer should have 2 * action_size outputs

        self.net = MLP([obs_size] + ([num_units] * num_layers) + [2*action_size])

        # store action scale and bias: the actor's output can be squashed to [-1, 1]
        self.action_scale = (action_high - action_low) / 2
        self.action_bias = (action_high + action_low) / 2

    def forward(self, x):
        #####################################################################
        # TODO: code the forward pass
        # 1. Pass the observation x through the network
        net_out = self.net(x)

        # 2. Split the output into mean (mu) and log_std
        #    (Hint: use torch.chunk(..., 2, dim=-1))
        mu, log_std = torch.chunk(net_out, 2, dim=-1)

        # 3. Constrain log_std to be within [LOG_STD_MIN, LOG_STD_MAX]
        #    (Hint: use torch.clamp)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        # 4. Calculate std = exp(log_std)
        std = torch.exp(log_std)

        # 5. Return a Normal distribution: torch.distributions.Normal(mu, std)
        
        # Return a distribution
        return torch.distributions.Normal(mu, std)


class Agent:

    # automatically select compute device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    buffer_size: int = 50_000  # no need to change

    #########################################################################
    # TODO: store and tune hyperparameters here

    batch_size: int = 256
    gamma: float = 0.99  # MDP discount factor

    tau: float = 0.005  # Polyak averaging coefficient
    learning_rate_q: float = 3e-4  # Learning rate for critics
    learning_rate_pi: float = 1e-4  # Learning rate for actor
    learning_rate_eta: float = 1e-3 # Learning rate for temperature eta
    target_kl: float = 0.01  # Target KL divergence for the M-step
    num_samples_q: int = 20  # Number of samples for E-step (critic update)
    num_samples_pi: int = 20 # Number of samples for M-step (actor update)
    
    #########################################################################

    def __init__(self, env):

        # extract informations from the environment
        self.obs_size = np.prod(env.observation_space.shape)  # size of observations
        self.action_size = np.prod(env.action_space.shape)  # size of actions
        # extract bounds of the action space
        self.action_low = torch.tensor(env.action_space.low).float()
        self.action_high = torch.tensor(env.action_space.high).float()
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2

        #####################################################################
        # TODO: initialize actor, critic and attributes
        # 1. Initialize Actor (pi) and Target Actor (pi_target)
        # (Use the Actor class)
        self.pi = Actor(
            action_high=self.action_high,
            action_low=self.action_low,
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=20,
            num_units=10
        )

        self.pi_target = Actor(
            action_high=self.action_high,
            action_low=self.action_low,
            obs_size=self.obs_size,
            action_size=self.action_size,
            num_layers=20,
            num_units=10
        )
        
        # 2. Initialize TWO Critics (q1, q2) and their Targets (q1_target, q2_target)
        # (This is the "twin critics" trick from TD3, which MPO also uses)
        self.q1 = Critic(
            action_size=self.action_size,
            obs_size=self.obs_size,
            num_layers=20,
            num_units=10
        )

        self.q2 = Critic(
            action_size=self.action_size,
            obs_size=self.obs_size,
            num_layers=20,
            num_units=10
        )

        self.q1_target = Critic(
            action_size=self.action_size,
            obs_size=self.obs_size,
            num_layers=20,
            num_units=10
        )

        self.q2_target = Critic(
            action_size=self.action_size,
            obs_size=self.obs_size,
            num_layers=20,
            num_units=10
        )

        # 3. Initialize the optimizers for q1, q2, and pi
        

        # 4. Initialize the learnable temperature 'eta' for the KL constraint
        self.log_eta = torch.tensor(1.0, device=self.device, requires_grad=True)
        self.eta_optimizer = optim.Adam([self.log_eta], lr=self.learning_rate_eta)

        #####################################################################
        # create buffer
        self.buffer = ReplayBuffer(self.buffer_size, self.obs_size, self.action_size, self.device)
        self.train_step = 0
    
    def train(self):
        '''
        Updates actor and critic with one batch from the replay buffer.
        '''
        obs, action, next_obs, done, reward = self.buffer.sample(self.batch_size)

        #####################################################################
        # TODO: code MPO training logic (E-Step and M-Step)

        with torch.no_grad():
            # 1. Get target policy distribution at next_obs
            # dist_target = self.pi_target(next_obs)
            dist_target = None  # Placeholder

            # 2. Sample K actions from the target policy distribution
            # (Hint: use .sample((self.num_samples_q,))
            
            # 3. Squash and rescale the K sampled actions
            
            # 4. Expand next_obs to match the K samples
            
            # 5. Get target Q-values for all (next_obs_expanded, next_actions_samples) pairs
            # (Use the twin target critics and take the minimum q_target_all = torch.min(self.q1_target(...), self.q2_target(...))

            # 6. Average the K target Q-values to get the expected Q-value (q_next)
            # (Hint: reshape q_target_all to [num_samples, batch_size, 1] and take mean(dim=0))
            
            # 7. Compute the final Bellman target (y)
            y = reward + (1 - done) * self.gamma * q_next

        # 8. Compute the critic loss (MSE loss) for BOTH critics
        q1_values = None
        q2_values = None
        q1_loss = None
        q2_loss = None
        q_loss = q1_loss + q2_loss
        
        # 9. Optimize the critic(s)
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # --- Policy Improvement (M-Step): Update Actor & Eta ---
        # Goal: Find a new policy that maximizes Q-values,
        #       subject to a KL constraint.
        
        # 10. Get online policy distribution at obs
        dist_online = None

        # 11. Sample K actions from the online policy distribution
        # (Use .rsample() for reparameterization)

        # 12. Squash and rescale the K sampled actions

        # 13. Expand obs to match the K samples
        
        # 14. Get Q-values for the (obs_expanded, actions_samples) pairs
        # (Use one of the online critics, e.g., q1. Detach from critic graph)
        
        # 15. Get the temperature eta
        eta = torch.exp(self.log_eta).detach()

        # 16. Compute the importance weights (w)
        # (Hint: w = softmax(q_values_samples / eta, dim=0))
        
        # 17. Compute the log-probability of the sampled actions
        #     under the online policy
        # (This requires care with the Tanh squashing)
        # log_prob_samples = ...
        
        # 18. Compute the policy loss (weighted log-likelihood)
        # (Hint: pi_loss = - (w * log_prob_samples).mean())
        
        # 19. Optimize the actor
        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        self.pi_optimizer.step()
        
        # --- Temperature (eta) Update ---

        # 20. Get the new online distribution (post-update)
        dist_online_new = self.pi(obs)
        kl = torch.distributions.kl_divergence(dist_online, dist_online_new.detach()).mean()
        
        # 21. Compute the temperature loss
        # (Hint: eta_loss = torch.exp(self.log_eta) * (self.target_kl - kl).detach())
        
        # 22. Optimize eta
        self.eta_optimizer.zero_grad()
        eta_loss.backward()
        self.eta_optimizer.step()

        # --- Target Network Updates ---

        # 23. Softly update all target networks (q1_target, q2_target, pi_target)
        # (Use Polyak averaging with self.tau)
        
        #####################################################################

    def get_action(self, obs, train):
        '''
        Returns the agent's action for a given observation.
        The train parameter can be used to control stochastic behavior.
        '''
        #####################################################################
        # TODO: return the agent's action
        
        # 1. Convert obs to a tensor and send to device

        # 2. Get the action distribution from the actor (self.pi)

        with torch.no_grad():
            # 3. Get action
            if train:
                # During training: sample from the distribution
                # (Hint: use dist.rsample() for reparameterization trick)
                action_gaussian = torch.zeros(1, self.action_size, device=self.device) # Placeholder
            else:
                # During testing: use the deterministic mean
                action_gaussian = torch.zeros(1, self.action_size, device=self.device) # Placeholder (use dist.mean)
            
            # 4. Squash the action through a Tanh function
            action_tanh = torch.zeros_like(action_gaussian) # Placeholder
            
            # 5. Rescale and shift the action to the environment's bounds
            # (Use self.action_scale and self.action_bias)
            action_scaled = torch.zeros_like(action_tanh) # Placeholder
            
            # 6. Convert to numpy array and return
            action = action_scaled.squeeze(0).cpu().numpy()
        
        #####################################################################
        return action

    def store(self, transition):
        '''
        Stores the observed transition in a replay buffer containing all past memories.
        '''
        obs, action, reward, next_obs, terminated = transition
        self.buffer.store(obs, next_obs, action, reward, terminated)


# This main function is provided here to enable some basic testing. 
# ANY changes here WON'T take any effect while grading.
if __name__ == '__main__':

    WARMUP_EPISODES = 10  # initial episodes of uniform exploration
    TRAIN_EPISODES = 50  # interactive episodes
    TEST_EPISODES = 300  # evaluation episodes
    save_video = False
    verbose = True
    seeds = np.arange(10)  # seeds for public evaluation

    start = time.time()
    print(f'Running public evaluation.') 
    test_returns = {k: [] for k in seeds}

    for seed in seeds:

        # seeding to ensure determinism
        seed = int(seed)
        for fn in [random.seed, np.random.seed, torch.manual_seed]:
            fn(seed)
        torch.backends.cudnn.deterministic = True

        env = get_env()
        env.action_space.seed(seed)
        env.np_random, _ = seeding.np_random(seed)

        agent = Agent(env)

        for _ in range(WARMUP_EPISODES):
            run_episode(env, agent, mode='warmup', verbose=verbose, rec=False)

        for _ in range(TRAIN_EPISODES):
            run_episode(env, agent, mode='train', verbose=verbose, rec=False)

        for n_ep in range(TEST_EPISODES):
            video_rec = (save_video and n_ep == TEST_EPISODES - 1)  # only record last episode
            with torch.no_grad():
                episode_return = run_episode(env, agent, mode='test', verbose=verbose, rec=video_rec)
            test_returns[seed].append(episode_return)

    avg_test_return = np.mean([np.mean(v) for v in test_returns.values()])
    within_seeds_deviation = np.mean([np.std(v) for v in test_returns.values()])
    across_seeds_deviation = np.std([np.mean(v) for v in test_returns.values()])
    print(f'Score for public evaluation: {avg_test_return}')
    print(f'Deviation within seeds: {within_seeds_deviation}')
    print(f'Deviation across seeds: {across_seeds_deviation}')

    print("Time :", (time.time() - start)/60, "min")
