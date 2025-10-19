import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from MARL.arguments import parse_args
from MARL.replay_buffer import ReplayBuffer
from MARL.train_util import *
from envs.myenv import MyEnv


def train(arglist):
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

    memory = ReplayBuffer(arglist.memory_size)
    writer = SummaryWriter(
        log_dir=f"runs/tensorboard/MADDPG_dispatch_{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}"
    )

    print("初始化智能体")
    env_agent = arglist.num_agents
    obs_shape_n = [state_dim for _ in range(env_agent)]
    action_shape_n = [action_dim for _ in range(env_agent)]

    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = get_trainers(
        env_agent, obs_shape_n, action_shape_n, arglist
    )

    action_size = []
    obs_size = []
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o += obs_shape
        end_a += action_shape
        obs_size.append((head_o, end_o))
        action_size.append((head_a, end_a))
        head_o = end_o
        head_a = end_a

    print("开始训练.......")
    start_time = time.time()
    game_step = 0
    model_update_cnt = 0

    try:
        for episode_gone in range(arglist.max_episode):
            plane_obs, defence_obs = env.reset()
            sum_episode_reward = 0.0
            agent_episode_reward = [0.0] * env_agent

            for episode_step in range(arglist.max_step):
                defence_obs_array = np.stack(defence_obs)
                defence_obs_tensor = torch.from_numpy(defence_obs_array).to(arglist.device, torch.float)

                defence_action_n = [agent(obs) for agent, obs in zip(actors_cur, defence_obs_tensor)]
                defence_action_n_2d = torch.stack(defence_action_n)
                defence_action_n = defence_action_n_2d.detach().cpu().numpy()

                new_plane_obs, new_defence_obs, rew_n, done_n, info = env.step(None, defence_action_n)

                memory.add(defence_obs, np.concatenate(defence_action_n), rew_n, new_defence_obs, done_n)

                agent_episode_reward = np.add(rew_n, agent_episode_reward)
                sum_episode_reward += float(np.sum(rew_n))

                plane_obs = new_plane_obs
                defence_obs = new_defence_obs
                game_step += 1

            model_update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(
                arglist,
                game_step,
                model_update_cnt,
                memory,
                obs_size,
                action_size,
                actors_cur,
                actors_tar,
                critics_cur,
                critics_tar,
                optimizers_a,
                optimizers_c,
            )
            if episode_gone == arglist.max_episode - 1:
                save_model_now(arglist, actors_cur, actors_tar, critics_cur, critics_tar)
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
            save_model_now(arglist, actors_cur, actors_tar, critics_cur, critics_tar)


def eval(arglist, online_render):
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
    actors_cur = get_eval_actor(env_agent, arglist)

    print("开始评估.......")
    for episode_gone in range(arglist.eval_episode):
        sum_episode_reward = 0.0
        agent_episode_reward = [0.0] * env_agent
        plane_obs, defence_obs = env.reset()
        print("Episode ", episode_gone)

        for episode_step in range(arglist.max_step):
            defence_obs_array = np.stack(defence_obs)
            defence_obs_tensor = torch.from_numpy(defence_obs_array).to(arglist.device, torch.float)
            defence_action_n = [agent(obs) for agent, obs in zip(actors_cur, defence_obs_tensor)]
            defence_action_n_2d = torch.stack(defence_action_n)
            defence_action = defence_action_n_2d.detach().cpu().numpy()

            _, new_defence_obs, rew_n, done_n, info = env.step(None, defence_action)
            agent_episode_reward = np.add(rew_n, agent_episode_reward)
            sum_episode_reward += float(np.sum(rew_n))

            if online_render:
                env.render()
                time.sleep(0.01)

            defence_obs = new_defence_obs
            plane_obs = plane_obs  # placeholder, plane dynamics are handled inside the environment

        print("=Training episode: ", episode_gone, "    episode reward: ", agent_episode_reward,
              " sum: ", sum_episode_reward / env_agent)


if __name__ == "__main__":
    arglist = parse_args()
    if arglist.train:
        train(arglist)
    else:
        eval(arglist, False)
