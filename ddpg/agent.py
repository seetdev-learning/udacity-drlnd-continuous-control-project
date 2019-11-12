import numpy as np
import random
import copy

from ddpg.model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(
        self, device, memory, config
    ):
        """Initialize an Agent object.
        
        Params
        ======
            device (object): hardware device to run on CPU or GPU
            memory (object): memory for replay buffer
            config (dict)
                - "state_size": dimension of each state
                - "action_size": dimension of each action
                - "buffer_size": replay buffer size
                - "batch_size": minibatch size
                - "random_seed": random seed
                - "gamma": discount factor
                - "tau": for soft update of target parameters
                - "lr_actor": learning rate of the actor 
                - "lr_critic": learning rate of the critic
                - "weight_decay": L2 weight decay
                - "learn_every": learn from replay buffer every time step
                - "learn_batch_size": number of batches to learn from replay buffer every learn_every time step
                - "grad_clip": gradient value to clip at for critic
                - "eps_start": starting value of epsilon, for epsilon-greedy action selection
                - "eps_end": minimum value of epsilon
                - "eps_decay": multiplicative factor (per episode) for decreasing epsilon
                - "print_every": Print average every x episode,
                - "episode_steps": Maximum number of steps to run for each episode
                - "mu": mu for noise
                - "theta": theta for noise 
                - "sigma": sigma for noise
        """
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        if config['random_seed'] is not  None:
            self.seed = random.seed(config['random_seed'])
        else:
            self.seed = random.seed()          
        self.eps = config['eps_start']
        self.eps_decay = config['eps_decay']
        self.eps_end = config['eps_end']

        self.device = device
        # Replay memory
        self.memory = memory
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.lr_actor = config['lr_actor']
        self.lr_critic = config['lr_critic']
        self.weight_decay = config['weight_decay']
        self.learn_every = config['learn_every']
        self.learn_batch_size = config['learn_batch_size']
        self.grad_clip = config['grad_clip']
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, config['random_seed']).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size, config['random_seed']).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size, config['random_seed']).to(self.device)
        self.critic_target = Critic(self.state_size, self.action_size, config['random_seed']).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr= self.lr_critic, weight_decay=self.weight_decay)

        # Noise process
        self.noise = OUNoise(config)
    
    def step(self, state, action, reward, next_state, done, timestep):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory and every update_every time steps
        if len(self.memory) > self.batch_size and timestep % self.learn_every == 0:
            for i in range(self.learn_batch_size):
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.eps * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()
    
    def learn_episode(self):
        self.learn(self.memory.sample(True))
    
    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # gradient clipping for critic
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target) 
        
        if self.eps_decay > 0:
            self.eps = max(self.eps_end, self.eps - self.eps_decay)             # decrease epsilon
            self.reset()
        
    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, config):
        """Initialize parameters and noise process."""
        self.mu = config['mu'] * np.ones(config['action_size'])
        self.theta =  config['theta']
        self.sigma = config['sigma']
        if config['random_seed'] is not None:
            self.seed = random.seed(config['random_seed'])
        else:
            self.seed = random.seed()         
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state