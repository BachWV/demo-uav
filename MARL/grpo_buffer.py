import numpy as np
import torch


class GRPOBuffer:
    """
    Buffer for GRPO algorithm to store trajectories for group-based training
    Stores full episodes for on-policy learning
    """
    def __init__(self, max_episodes, max_steps, num_agents, obs_dim, action_dim, device):
        """
        Initialize GRPO buffer
        Args:
            max_episodes: maximum number of episodes to store
            max_steps: maximum steps per episode
            num_agents: number of agents
            obs_dim: observation dimension
            action_dim: action dimension (1 for discrete)
            device: torch device
        """
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Storage for episodes
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.episode_lengths = []
        
        self.current_episode_obs = [[] for _ in range(num_agents)]
        self.current_episode_actions = [[] for _ in range(num_agents)]
        self.current_episode_log_probs = [[] for _ in range(num_agents)]
        self.current_episode_rewards = [[] for _ in range(num_agents)]
        self.current_episode_dones = [[] for _ in range(num_agents)]
        self.current_episode_values = [[] for _ in range(num_agents)]
        
        self.ptr = 0
        self.size = 0
    
    def add_step(self, obs, actions, log_probs, rewards, dones, values):
        """
        Add a step to current episode
        Args:
            obs: list of observations for each agent
            actions: list of actions for each agent
            log_probs: list of log probabilities for each agent
            rewards: list of rewards for each agent
            dones: list of done flags for each agent
            values: list of value estimates for each agent
        """
        for i in range(self.num_agents):
            self.current_episode_obs[i].append(obs[i])
            self.current_episode_actions[i].append(actions[i])
            self.current_episode_log_probs[i].append(log_probs[i])
            self.current_episode_rewards[i].append(rewards[i])
            self.current_episode_dones[i].append(dones[i])
            self.current_episode_values[i].append(values[i])
    
    def finish_episode(self):
        """
        Finish current episode and store in buffer
        """
        if len(self.current_episode_obs[0]) == 0:
            return
        
        episode_length = len(self.current_episode_obs[0])
        self.episode_lengths.append(episode_length)
        
        # Convert lists to numpy arrays
        obs_array = np.array([self.current_episode_obs[i] for i in range(self.num_agents)])
        actions_array = np.array([self.current_episode_actions[i] for i in range(self.num_agents)])
        log_probs_array = np.array([self.current_episode_log_probs[i] for i in range(self.num_agents)])
        rewards_array = np.array([self.current_episode_rewards[i] for i in range(self.num_agents)])
        dones_array = np.array([self.current_episode_dones[i] for i in range(self.num_agents)])
        values_array = np.array([self.current_episode_values[i] for i in range(self.num_agents)])
        
        if self.size < self.max_episodes:
            self.observations.append(obs_array)
            self.actions.append(actions_array)
            self.log_probs.append(log_probs_array)
            self.rewards.append(rewards_array)
            self.dones.append(dones_array)
            self.values.append(values_array)
            self.size += 1
        else:
            # Overwrite oldest episode
            self.observations[self.ptr] = obs_array
            self.actions[self.ptr] = actions_array
            self.log_probs[self.ptr] = log_probs_array
            self.rewards[self.ptr] = rewards_array
            self.dones[self.ptr] = dones_array
            self.values[self.ptr] = values_array
            self.episode_lengths[self.ptr] = episode_length
        
        self.ptr = (self.ptr + 1) % self.max_episodes
        
        # Clear current episode
        self.current_episode_obs = [[] for _ in range(self.num_agents)]
        self.current_episode_actions = [[] for _ in range(self.num_agents)]
        self.current_episode_log_probs = [[] for _ in range(self.num_agents)]
        self.current_episode_rewards = [[] for _ in range(self.num_agents)]
        self.current_episode_dones = [[] for _ in range(self.num_agents)]
        self.current_episode_values = [[] for _ in range(self.num_agents)]
    
    def get_all_data(self):
        """
        Get all stored data for training
        Returns:
            Dictionary containing all stored trajectories
        """
        if self.size == 0:
            return None
        
        return {
            'observations': self.observations[:self.size],
            'actions': self.actions[:self.size],
            'log_probs': self.log_probs[:self.size],
            'rewards': self.rewards[:self.size],
            'dones': self.dones[:self.size],
            'values': self.values[:self.size],
            'episode_lengths': self.episode_lengths[:self.size]
        }
    
    def clear(self):
        """
        Clear all stored data
        """
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.episode_lengths = []
        self.current_episode_obs = [[] for _ in range(self.num_agents)]
        self.current_episode_actions = [[] for _ in range(self.num_agents)]
        self.current_episode_log_probs = [[] for _ in range(self.num_agents)]
        self.current_episode_rewards = [[] for _ in range(self.num_agents)]
        self.current_episode_dones = [[] for _ in range(self.num_agents)]
        self.current_episode_values = [[] for _ in range(self.num_agents)]
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size
