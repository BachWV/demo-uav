import time
import torch
import argparse

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')


def parse_args():
    parser = argparse.ArgumentParser("MARL")

    # environment

    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True,
                        help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=20, help="Number of torch threads for training")
    parser.add_argument('--num_agents', type=int, default=12, help="number of players")
    parser.add_argument('--scenario_name', type=str, default="weight01", help="weight set")
    parser.add_argument("--note",type=str,default="双雷达还原奖励函数，增加对x轴倒飞的惩罚")

    parser.add_argument("--use_render", action='store_true', default=True,
                        help="by default, do not render-weight001-60% the env during training. If set, start render-weight001-60%. Note: something, the environment has internal render-weight001-60% process which is not controlled by this hyperparam.")
    parser.add_argument("--trace_dir", type=str, default="MADDPG3-1-radar")
    parser.add_argument("--train", type=bool, default=False)  # True False
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--save_dir", type=str, default="models/",
                        help="directory in which training state and model are loaded")
    parser.add_argument("--plane_model_name", type=str, default="models/2024-06-20-11-53-25/",
                        help="无人机集群模型的路径")
    parser.add_argument("--old_model_name", type=str, default="models/2024-06-20-11-53-25/",
                        help="the number of the episode for saving the model")
    parser.add_argument("--log_interval", type=int, default=50,
                        help="time duration between contiunous twice log printing.")



    # training parameters
    parser.add_argument("--device", default=device, help="torch device")
    parser.add_argument("--learning_start_step", type=int, default=10000, help="learning start steps")  # 50
    parser.add_argument("--learning_fre", type=int, default=200, help="learning frequency")  # 5
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--tao", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=1e-3, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=1e-3, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=200, help="number of episodes to optimize at the same time")
    parser.add_argument("--memory_size", type=int, default=1e6, help="number of data storedin the memory")
    parser.add_argument("--num_units_openai0", type=int, default=512, help="number of units in the mlp")
    parser.add_argument("--num_units_openai1", type=int, default=256, help="number of units in the mlp")
    parser.add_argument("--num_units_openai2", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_openai3", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--max_episode", type=int, default=120000, help="number of train")
    parser.add_argument("--eval_episode", type=int, default=100, help="number of eval")
    parser.add_argument("--max_step", type=int, default=250, help="max number of step")
    parser.add_argument("--norm", type=bool, default=False)
    parser.add_argument("--reward_norm", type=bool, default=False)




    # check pointing
    parser.add_argument("--fre_save_model", type=int, default=2500,
                        help="the number of the episode for saving the model")
    parser.add_argument("--start_save_model", type=int, default=400,
                        help="the number of the episode for saving the model")
    parser.add_argument("--save_model", type=int, default=1,
                        help="the number of the episode for saving the model")  # 40




    return parser.parse_args()
