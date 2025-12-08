import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()

    def act(self, input):
        policy, value = self.forward(input)
        return policy, value


class GRPO_Critic(abstract_agent):
    """
    GRPO Critic Network - estimates state value for advantage calculation
    """
    def __init__(self, obs_shape_n, args):
        super(GRPO_Critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(obs_shape_n, args.num_units_openai0)
        self.linear_c2 = nn.Linear(args.num_units_openai0, args.num_units_openai1)
        self.linear_c3 = nn.Linear(args.num_units_openai1, args.num_units_openai2)
        self.linear_c4 = nn.Linear(args.num_units_openai2, args.num_units_openai3)
        self.linear_c = nn.Linear(args.num_units_openai3, 1)
        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input):
        x = self.LReLU(self.linear_c1(obs_input))
        x = self.LReLU(self.linear_c2(x))
        x = self.LReLU(self.linear_c3(x))
        x = self.LReLU(self.linear_c4(x))
        value = self.linear_c(x)
        return value


class GRPO_Actor(abstract_agent):
    """
    GRPO Actor Network - outputs action probabilities for discrete actions
    """
    def __init__(self, num_inputs, action_size, args):
        super(GRPO_Actor, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_openai0)
        self.linear_a2 = nn.Linear(args.num_units_openai0, args.num_units_openai1)
        self.linear_a3 = nn.Linear(args.num_units_openai1, args.num_units_openai2)
        self.linear_a4 = nn.Linear(args.num_units_openai2, args.num_units_openai3)
        self.linear_a = nn.Linear(args.num_units_openai3, action_size)
        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a3.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a4.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, input, return_logits=False):
        """
        Forward pass
        Args:
            input: observation tensor
            return_logits: if True, return both logits and probabilities
        Returns:
            action probabilities (and optionally logits)
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        x = self.LReLU(self.linear_a3(x))
        x = self.LReLU(self.linear_a4(x))
        logits = self.linear_a(x)
        
        # Use softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        if return_logits:
            return logits, probs
        return probs
    
    def sample_action(self, obs, deterministic=False):
        """
        Sample action from policy
        Args:
            obs: observation tensor
            deterministic: if True, select argmax action
        Returns:
            action, log_prob, entropy
        """
        logits, probs = self.forward(obs, return_logits=True)
        dist = Categorical(probs)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate_actions(self, obs, actions):
        """
        Evaluate log probability and entropy of given actions
        Args:
            obs: observation tensor
            actions: action tensor
        Returns:
            log_probs, entropy
        """
        logits, probs = self.forward(obs, return_logits=True)
        dist = Categorical(probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy
