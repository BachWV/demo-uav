import torch

from MARL.replay_buffer import ReplayBuffer
from MARL.arguments import parse_args
from MARL.train_util import *
from torch.utils.tensorboard import SummaryWriter
from envs.myenv import MyEnv


def train(arglist):
    if torch.cuda.is_available():
        print("choose to use gpu...")
        torch.set_num_threads(arglist.n_training_threads)
        if arglist.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    # seed #随机数种子
    torch.manual_seed(arglist.seed)
    torch.cuda.manual_seed_all(arglist.seed)
    np.random.seed(arglist.seed)

    # 1、创建环境
    env = MyEnv(args_all=arglist)
    print("创建环境")

    state_dim = env.obs_dim
    action_dim = env.action_dim

    memory = ReplayBuffer(arglist.memory_size)  # 回放缓冲区
    writer = SummaryWriter(log_dir='runs/tensorboard/MADDPG_one_radar_{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))

    # 2、创建agent # 创建强化学习需要学习的模型
    print("创建智能体")
    env_agent = arglist.num_agents
    obs_shape_n = [state_dim for _ in range(env_agent)]  # 观测空间
    action_shape_n = [action_dim for _ in range(env_agent)]  # 动作空间

    actors_plane_cur = get_plane_actor(env_agent, arglist)

    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = get_trainers(
        env_agent, obs_shape_n, action_shape_n, arglist)

    action_size = []
    obs_size = []
    game_step = 0
    model_update_cnt = 0
    episode_rewards = [0.0]  # sum of rewards for all agents
    # agent_rewards = [[0.0] for _ in range(env_agent)]

    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a

    print("开始训练.......")
    start_time = time.time()
    try:
        for episode_gone in range(arglist.max_episode):
            defence_obs_n,obs_n = env.reset()
            sum_episode_reward = 0
            agent_episode_reward = [0.0] * env_agent
            for episode_step in range(arglist.max_step):
                # action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() for
                #             agent, obs in zip(actors_cur, obs_n)]
                obs_n_tensor = torch.from_numpy(np.array(obs_n)).to(arglist.device, torch.float)

                action_n = [agent(obs) for
                            agent, obs in zip(actors_plane_cur, obs_n_tensor)]

                action_n_2d = torch.stack(action_n)
                action_n = action_n_2d.detach().cpu().numpy()


                defence_obs_n_tensor = torch.from_numpy(np.array(defence_obs_n)).to(arglist.device, torch.float)

                defence_action_n = [agent(obs) for
                            agent, obs in zip(actors_cur, defence_obs_n_tensor)]

                defence_action_n_2d = torch.stack(defence_action_n)
                defence_action_n = defence_action_n_2d.detach().cpu().numpy()

                new_plane_obs_n, new_obs_n, rew_n, done_n, info = env.step(action_n, defence_action_n)

                # save the experience
                memory.add(obs_n, np.concatenate(action_n), rew_n, new_obs_n, done_n)

                #episode_rewards.append(np.sum(rew_n))
                agent_episode_reward = np.add(rew_n, agent_episode_reward)
                sum_episode_reward += np.sum(rew_n)  # np.sum(rew_n)

                # 基于新的观测做预测
                obs_n = new_obs_n
                game_step += 1

            # train our agents
            model_update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train(
                arglist, game_step, model_update_cnt,
                memory, obs_size, action_size,
                actors_cur, actors_tar,
                critics_cur, critics_tar,
                optimizers_a, optimizers_c)
            if episode_gone == arglist.max_episode - 1:
                save_model_now(arglist, actors_cur, actors_tar, critics_cur, critics_tar)
            writer.add_scalar('train_episode_rewards', sum_episode_reward/env_agent, global_step=episode_gone)
            print('=Training episode: ', episode_gone, '    episode reward: ', agent_episode_reward)

            if (episode_gone + 1) % arglist.log_interval == 0:
                end_time = time.time()
                minute = float(end_time - start_time) / 60
                print(
                    f'每轮{arglist.log_interval}局,本轮用时{minute}分钟，预计剩余时间{minute * (arglist.max_episode - episode_gone - 1) / arglist.log_interval}分钟')
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
    # seed #随机数种子
    torch.manual_seed(arglist.seed)
    torch.cuda.manual_seed_all(arglist.seed)
    np.random.seed(arglist.seed)

    # 1、创建环境
    env = MyEnv(args_all=arglist)
    print("创建环境")


    # 2、创建agent # 创建强化学习需要学习的模型
    print("创建智能体")
    env_agent = arglist.num_agents
    actors_cur = get_eval_actor(env_agent, arglist)


    print("开始评估.......")
    for episode_gone in range(arglist.eval_episode):
        sum_episode_reward = 0
        agent_episode_reward = [0.0] * env_agent
        obs_n = env.reset()
        print('Episode  ', episode_gone)
        for episode_step in range(arglist.max_step):
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() for
                        agent, obs in zip(actors_cur, obs_n)]

            new_obs_n, rew_n, done_n, info = env.step(action_n)
            agent_episode_reward = np.add(rew_n, agent_episode_reward)
            sum_episode_reward += np.sum(rew_n)  # np.sum(rew_n)

            if online_render:
                env.render()
                time.sleep(0.01)
            # 基于新的观测做预测
            obs_n = new_obs_n

        print('=Training episode: ', episode_gone, '    episode reward: ', agent_episode_reward,' sum: ',sum_episode_reward/env_agent)


if __name__ == '__main__':
    arglist = parse_args()  # 参数集合
    if arglist.train:
        train(arglist)
    else:
        eval(arglist, False)

