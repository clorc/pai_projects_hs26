"""
This file contains the Cartpole environment, as well as couple of useful functions,
which you can use for the assignment.

IMPORTANT NOTE: CHANGING THIS FILE OR YOUR LOCAL EVALUATION MIGHT NOT WORK. CHANGING THIS FILE WON'T
AFFECT YOUR SUBMISSION RESULT IN THE CHECKER. 

"""
import torch
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit


class CustomCartpole(CartPoleEnv):
    """
    Modified cartpole environment. With respect to the gymnasium implementation:
    - actions are continuous instead of discrete
    - the episode never terminates
    """
    def __init__(self, render_mode='rgb_array', **kwargs):
        super().__init__(render_mode = 'rgb_array', *kwargs)
        self.action_space = spaces.Box(-np.ones(1), np.ones(1), dtype=np.float64)
        # self.gravity = 20  # uncomment to make it a little harder :)

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag * action.item()
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * np.square(theta_dot) * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        reward = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        ) * -1.

        if self.render_mode == "human":
            self.render()

        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added in `get_env`
        # unlike the original environment, the episode does not terminate on failure
        return np.array(self.state, dtype=np.float32), reward, False, False, {}
    

class ReplayBuffer():
    '''
    This class implements a FIFO replay buffer for storing transitions.
    Transitions are stored one at a time. Batches of transitions can be
    sampled with the 'sample' method.
    '''
    def __init__(self, buffer_size, obs_size, action_size, device):
        self.observations = np.zeros((buffer_size, obs_size), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, obs_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros((buffer_size), dtype=np.float32)
        self.dones = np.zeros((buffer_size), dtype=np.float32)
        self.timeouts = np.zeros((buffer_size), dtype=np.float32)
        self.pos = 0
        self.curr_size = 0
        self.max_size = buffer_size
        self.device = device

    def store(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = np.array(obs).copy()
        self.next_observations[self.pos] = np.array(next_obs).copy()
        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()
        self.pos = (self.pos + 1) % self.max_size
        self.curr_size = min(self.curr_size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.curr_size, size=batch_size)
        batch = (
            self.observations[idxs],
            self.actions[idxs],
            self.next_observations[idxs],
            (self.dones[idxs] * (1 - self.timeouts[idxs])),
            self.rewards[idxs],
        )
        return (torch.tensor(e, device=self.device).float() for e in batch)


def get_env():
    '''
    This function returns the environment.
    '''
    return TimeLimit(CustomCartpole(render_mode='rgb_array'), max_episode_steps=200)


def run_episode(env, agent, mode, verbose=False, rec=False):
    '''
    This function runs one episode of environment interaction.
    Until the episode is not finished (200 steps), it samples and performs an action,
    stores the transition in the buffer and allows the agent to train.
    
    :param env: the environment to run the episode on
    :param agent: the agent to use for the episode
    :param mode: selects between warmup, train and test
    :param verbose: whether to print the episode return and mode
    :param rec: whether to render the episode in a video

    Returns:
    :return: the cumulative reward over the episode
    '''

    assert mode in ['warmup', 'train', 'test'], 'Unknown mode'

    if rec:
        from gymnasium.wrappers.record_video import RecordVideo
        env = RecordVideo(env, video_folder='.', name_prefix='policy')

    # reset and initialize environment
    obs, _ = env.reset()
    episode_return, done = 0.0, False

    while not done:

        if mode == 'warmup':
            # sample actions uniformly at random
            action = env.action_space.sample()
        else:
            # query the agent for an action
            with torch.no_grad():
                action = agent.get_action(obs, mode=='train')

        # simulate one step in the environment
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_return += reward

        if mode != 'test':
            # save experience in a replay buffer
            agent.store((obs, action, reward, next_obs, terminated))
        if mode == 'train':
            # train the agent
            agent.train()

        # important step: reassign the last observation
        obs = next_obs

    if verbose:
        print(f"{mode.capitalize()} return: {episode_return}")

    if rec:
        env.stop_recording()

    return episode_return
