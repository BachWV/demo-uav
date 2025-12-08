import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from MARL.arguments import parse_args
from MARL.grpo_buffer import GRPOBuffer
from MARL.train_util_grpo import get_grpo_trainers, get_eval_grpo_actor, grpo_update, save_grpo_model
from envs.myenv import MyEnv


def train_grpo(arglist):
    """
    Train agents using GRPO algorithm
    """
    if torch.cuda.is_available():
        print("choose to use gpu...")
        torch.set_num_threads(arglist.n_training_threads)
        if arglist.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    torch.manual_seed(arglist.seed)
    torch.cuda.manual_seed_all(arglist.seed)
    np.random.seed(arglist.seed)

    env = MyEnv(args_all=arglist)
    print("环境初始化完成")

    state_dim = env.obs_dim
    action_dim = env.action_dim

    # Create GRPO buffer
    buffer = GRPOBuffer(
        max_episodes=arglist.grpo_buffer_size,
        max_steps=arglist.max_step,
        num_agents=arglist.num_agents,
        obs_dim=state_dim,
        action_dim=1,  # discrete action
        device=arglist.device
    )

    writer = SummaryWriter(
        log_dir=f"runs/tensorboard/GRPO_dispatch_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    )

    print("初始化GRPO智能体")
    env_agent = arglist.num_agents
    obs_shape_n = [state_dim for _ in range(env_agent)]
    action_shape_n = [action_dim for _ in range(env_agent)]

    actors, critics, optimizers_a, optimizers_c = get_grpo_trainers(
        env_agent, obs_shape_n, action_shape_n, arglist
    )

    print("开始GRPO训练.......")
    start_time = time.time()
    episode_count = 0
    update_count = 0

    try:
        for episode_gone in range(arglist.max_episode):
            plane_obs, defence_obs = env.reset()
            sum_episode_reward = 0.0
            agent_episode_reward = [0.0] * env_agent

            for episode_step in range(arglist.max_step):
                # Convert observations to tensor
                defence_obs_array = np.stack(defence_obs)
                defence_obs_tensor = torch.from_numpy(defence_obs_array).to(arglist.device, torch.float)

                # Sample actions from policy
                actions_list = []
                log_probs_list = []
                values_list = []

                for agent_idx, (actor, critic, obs_t) in enumerate(zip(actors, critics, defence_obs_tensor)):
                    # Sample action
                    action, log_prob, _ = actor.sample_action(obs_t.unsqueeze(0), deterministic=False)
                    
                    # Get value estimate
                    value = critic(obs_t.unsqueeze(0))
                    
                    actions_list.append(action.item())
                    log_probs_list.append(log_prob.item())
                    values_list.append(value.item())

                # Convert to numpy for environment
                defence_action_n = np.array(actions_list)

                # Step environment
                new_plane_obs, new_defence_obs, rew_n, done_n, info = env.step(None, defence_action_n)

                # Store transition in buffer
                buffer.add_step(
                    obs=defence_obs,
                    actions=actions_list,
                    log_probs=log_probs_list,
                    rewards=rew_n,
                    dones=done_n,
                    values=values_list
                )

                agent_episode_reward = np.add(rew_n, agent_episode_reward)
                sum_episode_reward += float(np.sum(rew_n))

                plane_obs = new_plane_obs
                defence_obs = new_defence_obs

            # Finish episode
            buffer.finish_episode()
            episode_count += 1

            # Update policy every N episodes
            if episode_count % arglist.grpo_update_interval == 0:
                print(f"Updating GRPO policy at episode {episode_gone}...")
                
                # Get all data from buffer
                buffer_data = buffer.get_all_data()
                
                # Update for multiple epochs
                for epoch in range(arglist.grpo_epochs):
                    loss_stats = grpo_update(
                        arglist, buffer_data, actors, critics,
                        optimizers_a, optimizers_c
                    )
                    
                    if loss_stats is not None:
                        writer.add_scalar("grpo/actor_loss", loss_stats['actor_loss'], update_count)
                        writer.add_scalar("grpo/critic_loss", loss_stats['critic_loss'], update_count)
                        writer.add_scalar("grpo/entropy", loss_stats['entropy'], update_count)
                
                # Clear buffer after update
                buffer.clear()
                update_count += 1
                print(f"GRPO update {update_count} completed.")

            # Save model
            if episode_gone == arglist.max_episode - 1:
                save_grpo_model(arglist, actors, critics)
            
            # Log training progress
            writer.add_scalar("train_episode_rewards", sum_episode_reward / env_agent, global_step=episode_gone)
            print("=Training episode: ", episode_gone, "    episode reward: ", agent_episode_reward)

            if (episode_gone + 1) % arglist.log_interval == 0:
                end_time = time.time()
                minute = float(end_time - start_time) / 60
                remaining = minute * (arglist.max_episode - episode_gone - 1) / arglist.log_interval
                print(f"每{arglist.log_interval}轮, 本轮耗时{minute:.2f}分钟, 预计剩余{remaining:.2f}分钟")
                start_time = end_time

    except KeyboardInterrupt:
        str_input = input("y for save\n")
        if str_input.startswith("y"):
            save_grpo_model(arglist, actors, critics)


def eval_grpo(arglist, online_render):
    """
    Evaluate GRPO agents
    """
    if torch.cuda.is_available():
        print("choose to use gpu...")
        torch.set_num_threads(arglist.n_training_threads)
        if arglist.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    torch.manual_seed(arglist.seed)
    torch.cuda.manual_seed_all(arglist.seed)
    np.random.seed(arglist.seed)

    env = MyEnv(args_all=arglist)
    print("环境初始化完成")

    env_agent = arglist.num_agents
    actors = get_eval_grpo_actor(env_agent, arglist)

    print("开始GRPO评估.......")
    for episode_gone in range(arglist.eval_episode):
        sum_episode_reward = 0.0
        agent_episode_reward = [0.0] * env_agent
        plane_obs, defence_obs = env.reset()
        print("Episode ", episode_gone)

        for episode_step in range(arglist.max_step):
            defence_obs_array = np.stack(defence_obs)
            defence_obs_tensor = torch.from_numpy(defence_obs_array).to(arglist.device, torch.float)
            
            # Use deterministic actions for evaluation
            actions_list = []
            for agent_idx, (actor, obs_t) in enumerate(zip(actors, defence_obs_tensor)):
                action, _, _ = actor.sample_action(obs_t.unsqueeze(0), deterministic=True)
                actions_list.append(action.item())
            
            defence_action = np.array(actions_list)

            _, new_defence_obs, rew_n, done_n, info = env.step(None, defence_action)
            agent_episode_reward = np.add(rew_n, agent_episode_reward)
            sum_episode_reward += float(np.sum(rew_n))

            if online_render:
                env.render()
                time.sleep(0.01)

            defence_obs = new_defence_obs

        print("=Eval episode: ", episode_gone, "    episode reward: ", agent_episode_reward,
              " sum: ", sum_episode_reward / env_agent)


if __name__ == "__main__":
    arglist = parse_args()
    if arglist.train:
        train_grpo(arglist)
    else:
        eval_grpo(arglist, False)
