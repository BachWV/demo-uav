import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from MARL.Model_GRPO import GRPO_Actor, GRPO_Critic


def get_grpo_trainers(env_agent, obs_shape_n, action_shape_n, arglist):
    """
    Initialize GRPO actors and critics
    """
    actors = [None for _ in range(env_agent)]
    critics = [None for _ in range(env_agent)]
    optimizers_a = [None for _ in range(env_agent)]
    optimizers_c = [None for _ in range(env_agent)]

    if arglist.restore:
        # Load existing model
        for idx in range(env_agent):
            actors[idx] = torch.load(arglist.old_model_name + f'grpo_a_{idx}.pt', weights_only=False, map_location=arglist.device)
            critics[idx] = torch.load(arglist.old_model_name + f'grpo_c_{idx}.pt', weights_only=False, map_location=arglist.device)
            optimizers_a[idx] = optim.Adam(actors[idx].parameters(), arglist.lr_a)
            optimizers_c[idx] = optim.Adam(critics[idx].parameters(), arglist.lr_c)
        print("Load GRPO Model......")
    else:
        # Create new model
        for i in range(env_agent):
            actors[i] = GRPO_Actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            critics[i] = GRPO_Critic(obs_shape_n[i], arglist).to(arglist.device)
            optimizers_a[i] = optim.Adam(actors[i].parameters(), arglist.lr_a)
            optimizers_c[i] = optim.Adam(critics[i].parameters(), arglist.lr_c)
        print("Init GRPO Model......")

    return actors, critics, optimizers_a, optimizers_c


def get_eval_grpo_actor(env_agent, arglist):
    """
    Load GRPO actors for evaluation
    """
    actors = [None for _ in range(env_agent)]
    for idx in range(env_agent):
        actors[idx] = torch.load(arglist.old_model_name + f'grpo_a_{idx}.pt', weights_only=False, map_location=arglist.device)
    return actors


def compute_gae_advantages(rewards, values, dones, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation (GAE)
    Args:
        rewards: tensor of shape (batch_size, num_steps)
        values: tensor of shape (batch_size, num_steps)
        dones: tensor of shape (batch_size, num_steps)
        gamma: discount factor
        gae_lambda: GAE lambda parameter
    Returns:
        advantages, returns
    """
    batch_size, num_steps = rewards.shape
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    
    gae = 0
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_value = 0
        else:
            next_value = values[:, t + 1]
        
        delta = rewards[:, t] + gamma * next_value * (1 - dones[:, t]) - values[:, t]
        gae = delta + gamma * gae_lambda * (1 - dones[:, t]) * gae
        advantages[:, t] = gae
        returns[:, t] = advantages[:, t] + values[:, t]
    
    return advantages, returns


def compute_group_relative_advantages(advantages, group_indices):
    """
    Compute group-relative advantages for GRPO
    Args:
        advantages: tensor of advantages for all agents
        group_indices: list of agent indices in the same group
    Returns:
        relative_advantages: group-normalized advantages
    """
    # Normalize advantages within the group
    group_advantages = advantages[group_indices]
    mean_adv = group_advantages.mean()
    std_adv = group_advantages.std() + 1e-8
    relative_advantages = (group_advantages - mean_adv) / std_adv
    
    return relative_advantages


def grpo_update(arglist, buffer_data, actors, critics, optimizers_a, optimizers_c):
    """
    Update GRPO agents using collected trajectories
    Args:
        arglist: argument list
        buffer_data: dictionary containing trajectory data
        actors: list of actor networks
        critics: list of critic networks
        optimizers_a: list of actor optimizers
        optimizers_c: list of critic optimizers
    Returns:
        loss statistics
    """
    if buffer_data is None:
        return None
    
    observations = buffer_data['observations'] # List of (num_agents, steps, obs_dim)
    actions = buffer_data['actions'] # List of (num_agents, steps)
    old_log_probs = buffer_data['log_probs'] # List of (num_agents, steps)
    rewards = buffer_data['rewards'] # List of (num_agents, steps)
    dones = buffer_data['dones'] # List of (num_agents, steps)
    old_values = buffer_data['values'] # List of (num_agents, steps)
    
    num_episodes = len(observations)
    num_agents = len(actors)
    
    # 1. Pre-compute Advantages and Returns for ALL agents and ALL episodes
    # We do this BEFORE the epoch loop to keep targets stationary
    
    # Storage for processed data
    # Shape: (num_agents, num_episodes, steps)
    all_advantages = [] 
    all_returns = []
    
    # Process each agent to compute GAE
    for agent_idx in range(num_agents):
        agent_advantages = []
        agent_returns = []
        
        for ep_idx in range(num_episodes):
            ep_obs = observations[ep_idx][agent_idx]
            ep_rewards = rewards[ep_idx][agent_idx]
            ep_dones = dones[ep_idx][agent_idx]
            
            # Convert to tensors
            obs_tensor = torch.FloatTensor(ep_obs).to(arglist.device)
            rewards_tensor = torch.FloatTensor(ep_rewards).to(arglist.device).unsqueeze(0)
            dones_tensor = torch.FloatTensor(ep_dones).to(arglist.device).unsqueeze(0)
            
            # Compute values with current critic (no grad)
            with torch.no_grad():
                values = critics[agent_idx](obs_tensor).squeeze(-1).unsqueeze(0)
            
            # Compute advantages using GAE
            adv, ret = compute_gae_advantages(
                rewards_tensor, values, dones_tensor,
                arglist.gamma, arglist.gae_lambda
            )
            
            agent_advantages.append(adv.squeeze(0))
            agent_returns.append(ret.squeeze(0))
            
        all_advantages.append(torch.stack(agent_advantages))
        all_returns.append(torch.stack(agent_returns))
    
    # Stack all agents: (num_agents, num_episodes, steps)
    all_advantages = torch.stack(all_advantages)
    all_returns = torch.stack(all_returns)
    
    # 2. Group Relative Normalization
    # Normalize advantages across the agent dimension for each episode/step
    # Shape: (num_agents, num_episodes, steps)
    mean_adv = all_advantages.mean(dim=0, keepdim=True)
    std_adv = all_advantages.std(dim=0, keepdim=True) + 1e-8
    normalized_advantages = (all_advantages - mean_adv) / std_adv
    
    # 3. Optimization Loop (Epochs)
    total_actor_loss = 0
    total_critic_loss = 0
    total_entropy = 0
    
    for _ in range(arglist.grpo_epochs):
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        epoch_entropy = 0
        
        for agent_idx in range(num_agents):
            agent_actor_losses = []
            agent_critic_losses = []
            agent_entropies = []
            
            # Flatten episodes for batch processing
            # We can process all episodes for one agent in a single batch if memory allows
            # Or loop through episodes. Let's loop for simplicity and memory safety.
            
            for ep_idx in range(num_episodes):
                ep_obs = observations[ep_idx][agent_idx]
                ep_actions = actions[ep_idx][agent_idx]
                ep_old_log_probs = old_log_probs[ep_idx][agent_idx]
                
                # Get pre-computed targets
                ep_advantages = normalized_advantages[agent_idx, ep_idx]
                ep_returns = all_returns[agent_idx, ep_idx]
                
                # Convert inputs to tensors
                obs_tensor = torch.FloatTensor(ep_obs).to(arglist.device)
                actions_tensor = torch.LongTensor(ep_actions).to(arglist.device)
                old_log_probs_tensor = torch.FloatTensor(ep_old_log_probs).to(arglist.device)
                
                # Evaluate actions with current policy
                new_log_probs, entropy = actors[agent_idx].evaluate_actions(obs_tensor, actions_tensor)
                
                # Get current values
                current_values = critics[agent_idx](obs_tensor).squeeze(-1)
                
                # Compute ratio for PPO-style clipping
                ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                
                # Compute surrogate losses
                surr1 = ratio * ep_advantages
                surr2 = torch.clamp(ratio, 1.0 - arglist.clip_param, 1.0 + arglist.clip_param) * ep_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                actor_loss = actor_loss - arglist.entropy_coef * entropy.mean()
                
                # Critic loss
                critic_loss = nn.MSELoss()(current_values, ep_returns)
                
                agent_actor_losses.append(actor_loss)
                agent_critic_losses.append(critic_loss)
                agent_entropies.append(entropy.mean())
            
            # Update Agent
            avg_actor_loss = torch.stack(agent_actor_losses).mean()
            avg_critic_loss = torch.stack(agent_critic_losses).mean()
            
            optimizers_a[agent_idx].zero_grad()
            avg_actor_loss.backward()
            nn.utils.clip_grad_norm_(actors[agent_idx].parameters(), arglist.max_grad_norm)
            optimizers_a[agent_idx].step()
            
            optimizers_c[agent_idx].zero_grad()
            avg_critic_loss.backward()
            nn.utils.clip_grad_norm_(critics[agent_idx].parameters(), arglist.max_grad_norm)
            optimizers_c[agent_idx].step()
            
            epoch_actor_loss += avg_actor_loss.item()
            epoch_critic_loss += avg_critic_loss.item()
            epoch_entropy += torch.stack(agent_entropies).mean().item()
            
        total_actor_loss += epoch_actor_loss / num_agents
        total_critic_loss += epoch_critic_loss / num_agents
        total_entropy += epoch_entropy / num_agents
    
    return {
        'actor_loss': total_actor_loss / arglist.grpo_epochs,
        'critic_loss': total_critic_loss / arglist.grpo_epochs,
        'entropy': total_entropy / arglist.grpo_epochs
    }


def save_grpo_model(arglist, actors, critics):
    """
    Save GRPO model
    """
    print(arglist.note)
    model_file_dir = os.path.join(arglist.save_dir, 'GRPO_' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()))
    print(model_file_dir)
    
    if not os.path.exists(model_file_dir):
        os.makedirs(model_file_dir)
    
    txt_file_path = os.path.join(model_file_dir, "grpo_note.txt")
    with open(txt_file_path, 'w') as f:
        strlist = ("note:{}\n"
                   "old_model_name:{}\n"
                   "model_file_dir:{}").format(arglist.note, arglist.old_model_name, model_file_dir)
        f.write(strlist)
    
    for agent_idx, (actor, critic) in enumerate(zip(actors, critics)):
        torch.save(actor, os.path.join(model_file_dir, f'grpo_a_{agent_idx}.pt'))
        torch.save(critic, os.path.join(model_file_dir, f'grpo_c_{agent_idx}.pt'))
    
    print(f"GRPO model saved to {model_file_dir}")
