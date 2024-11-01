import torch

from MARL.replay_buffer import ReplayBuffer
from MARL.arguments import parse_args
from MARL.train_util import *
from torch.utils.tensorboard import SummaryWriter
from envs.myenv import MyEnv


class MADDPG(object):
    def __init__(self):
        arglist = parse_args()  # 参数集合
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
        self.env = MyEnv(args_all=arglist)
        print("创建环境")
        state_dim = self.env.obs_dim
        action_dim = self.env.action_dim

        # 2、创建agent # 创建强化学习需要学习的模型
        print("创建智能体")
        env_agent = arglist.num_agents
        obs_shape_n = [state_dim for _ in range(env_agent)]  # 观测空间
        action_shape_n = [action_dim for _ in range(env_agent)]  # 动作空间
        actors_cur = get_eval_actor(env_agent, arglist)

        action_size = []
        obs_size = []
        game_step = 0

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

    def update(self):
        obs_n = self.env.reset()
        for episode_step in range(arglist.max_step):
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() for
                        agent, obs in zip(actors_cur, obs_n)]

            new_obs_n, rew_n, done_n, info = env.step(action_n)

            if online_render:
                env.render()
                # env.draw_graph()
                time.sleep(0.01)
            # 基于新的观测做预测
            obs_n = new_obs_n


        

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

    state_dim = env.obs_dim
    action_dim = env.action_dim

    # 2、创建agent # 创建强化学习需要学习的模型
    print("创建智能体")
    env_agent = arglist.num_agents
    obs_shape_n = [state_dim for _ in range(env_agent)]  # 观测空间
    action_shape_n = [action_dim for _ in range(env_agent)]  # 动作空间
    actors_cur = get_eval_actor(env_agent, arglist)

    action_size = []
    obs_size = []
    game_step = 0

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

    # if not arglist.train:
    #     env.__init_graph__()
    start_time = time.time()
    print("开始评估.......")
    for episode_gone in range(arglist.eval_episode):
        obs_n = env.reset()
        print('Episode  ', episode_gone)
        for episode_step in range(arglist.max_step):
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() for
                        agent, obs in zip(actors_cur, obs_n)]

            new_obs_n, rew_n, done_n, info = env.step(action_n)

            if online_render:
                env.render()
                # env.draw_graph()
                time.sleep(0.01)
            # 基于新的观测做预测
            obs_n = new_obs_n

        # env.close_draw()
        # if (episode_gone + 1) % arglist.log_interval == 0:
        #     end_time = time.time()
        #     minute = float(end_time - start_time) / 60
        #     print(
        #         f'每轮{arglist.log_interval}局,本轮用时{minute}分钟，预计剩余时间{minute * (arglist.max_episode - episode_gone - 1) / arglist.log_interval}分钟')
        #     start_time = end_time


if __name__ == '__main__':
    arglist = parse_args()  # 参数集合
    if arglist.train:
        train(arglist)
    else:
        eval(arglist, True)
