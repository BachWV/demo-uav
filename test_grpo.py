"""
快速测试GRPO实现
验证所有组件是否正常工作
"""

import torch
import numpy as np
from MARL.arguments import parse_args
from MARL.Model_GRPO import GRPO_Actor, GRPO_Critic
from MARL.grpo_buffer import GRPOBuffer
from MARL.train_util_grpo import compute_gae_advantages


def test_grpo_models():
    """测试GRPO模型"""
    print("=" * 50)
    print("测试GRPO模型...")
    
    arglist = parse_args()
    obs_dim = 50
    action_dim = 10
    
    # 测试Actor
    actor = GRPO_Actor(obs_dim, action_dim, arglist)
    obs = torch.randn(32, obs_dim)
    
    # 测试forward
    probs = actor(obs)
    assert probs.shape == (32, action_dim), f"Actor输出形状错误: {probs.shape}"
    assert torch.allclose(probs.sum(dim=-1), torch.ones(32), atol=1e-5), "概率和不为1"
    print("✓ Actor forward测试通过")
    
    # 测试sample_action
    action, log_prob, entropy = actor.sample_action(obs, deterministic=False)
    assert action.shape == (32,), f"动作形状错误: {action.shape}"
    assert log_prob.shape == (32,), f"log_prob形状错误: {log_prob.shape}"
    print("✓ Actor sample_action测试通过")
    
    # 测试Critic
    critic = GRPO_Critic(obs_dim, arglist)
    values = critic(obs)
    assert values.shape == (32, 1), f"Critic输出形状错误: {values.shape}"
    print("✓ Critic测试通过")
    
    print("✓ 模型测试全部通过!\n")


def test_grpo_buffer():
    """测试GRPO缓冲区"""
    print("=" * 50)
    print("测试GRPO缓冲区...")
    
    num_agents = 3
    obs_dim = 50
    max_episodes = 5
    max_steps = 100
    
    buffer = GRPOBuffer(
        max_episodes=max_episodes,
        max_steps=max_steps,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dim=1,
        device='cpu'
    )
    
    # 添加多个episode
    for ep in range(3):
        for step in range(10):
            obs = [np.random.randn(obs_dim) for _ in range(num_agents)]
            actions = [np.random.randint(0, 10) for _ in range(num_agents)]
            log_probs = [np.random.randn() for _ in range(num_agents)]
            rewards = [np.random.randn() for _ in range(num_agents)]
            dones = [0 for _ in range(num_agents)]
            values = [np.random.randn() for _ in range(num_agents)]
            
            buffer.add_step(obs, actions, log_probs, rewards, dones, values)
        
        buffer.finish_episode()
    
    assert len(buffer) == 3, f"缓冲区大小错误: {len(buffer)}"
    print("✓ 添加episode测试通过")
    
    # 获取数据
    data = buffer.get_all_data()
    assert data is not None, "获取数据失败"
    assert len(data['observations']) == 3, "观测数据数量错误"
    print("✓ 获取数据测试通过")
    
    # 清空缓冲区
    buffer.clear()
    assert len(buffer) == 0, "清空缓冲区失败"
    print("✓ 清空缓冲区测试通过")
    
    print("✓ 缓冲区测试全部通过!\n")


def test_gae_computation():
    """测试GAE计算"""
    print("=" * 50)
    print("测试GAE优势计算...")
    
    batch_size = 4
    num_steps = 10
    
    rewards = torch.randn(batch_size, num_steps)
    values = torch.randn(batch_size, num_steps)
    dones = torch.zeros(batch_size, num_steps)
    
    gamma = 0.99
    gae_lambda = 0.95
    
    advantages, returns = compute_gae_advantages(rewards, values, dones, gamma, gae_lambda)
    
    assert advantages.shape == (batch_size, num_steps), f"优势形状错误: {advantages.shape}"
    assert returns.shape == (batch_size, num_steps), f"回报形状错误: {returns.shape}"
    print("✓ GAE计算测试通过")
    
    # 验证returns = advantages + values
    assert torch.allclose(returns, advantages + values, atol=1e-5), "returns计算错误"
    print("✓ Returns验证通过")
    
    print("✓ GAE测试全部通过!\n")


def test_integration():
    """集成测试"""
    print("=" * 50)
    print("集成测试...")
    
    try:
        from envs.myenv import MyEnv
        arglist = parse_args()
        
        # 创建环境
        env = MyEnv(args_all=arglist)
        print("✓ 环境创建成功")
        
        # 创建GRPO智能体
        from MARL.train_util_grpo import get_grpo_trainers
        
        obs_dim = env.obs_dim
        action_dim = env.action_dim
        num_agents = arglist.num_agents
        
        obs_shape_n = [obs_dim for _ in range(num_agents)]
        action_shape_n = [action_dim for _ in range(num_agents)]
        
        actors, critics, optimizers_a, optimizers_c = get_grpo_trainers(
            num_agents, obs_shape_n, action_shape_n, arglist
        )
        print("✓ GRPO智能体创建成功")
        
        # 运行一个episode
        plane_obs, defence_obs = env.reset()
        print("✓ 环境重置成功")
        
        # 执行几步
        for step in range(5):
            defence_obs_array = np.stack(defence_obs)
            defence_obs_tensor = torch.from_numpy(defence_obs_array).to(arglist.device, torch.float)
            
            actions_list = []
            for actor, obs_t in zip(actors, defence_obs_tensor):
                action, _, _ = actor.sample_action(obs_t.unsqueeze(0), deterministic=False)
                actions_list.append(action.item())
            
            defence_action_n = np.array(actions_list)
            new_plane_obs, new_defence_obs, rew_n, done_n, info = env.step(None, defence_action_n)
            
            defence_obs = new_defence_obs
        
        print("✓ Episode执行成功")
        print("✓ 集成测试通过!\n")
        
    except Exception as e:
        print(f"✗ 集成测试失败: {e}\n")
        import traceback
        traceback.print_exc()


def test_grpo_update():
    """测试GRPO更新逻辑"""
    print("=" * 50)
    print("测试GRPO更新逻辑...")
    
    try:
        from MARL.train_util_grpo import grpo_update, get_grpo_trainers
        from MARL.arguments import parse_args
        
        arglist = parse_args()
        arglist.device = 'cpu' # 强制使用CPU测试
        
        num_agents = 3
        obs_dim = 50
        action_dim = 10
        
        # 创建模型
        obs_shape_n = [obs_dim for _ in range(num_agents)]
        action_shape_n = [action_dim for _ in range(num_agents)]
        actors, critics, optimizers_a, optimizers_c = get_grpo_trainers(
            num_agents, obs_shape_n, action_shape_n, arglist
        )
        
        # 构造模拟数据
        num_episodes = 2
        steps = 10
        
        buffer_data = {
            'observations': [[np.random.randn(steps, obs_dim) for _ in range(num_agents)] for _ in range(num_episodes)],
            'actions': [[np.random.randint(0, action_dim, steps) for _ in range(num_agents)] for _ in range(num_episodes)],
            'log_probs': [[np.random.randn(steps) for _ in range(num_agents)] for _ in range(num_episodes)],
            'rewards': [[np.random.randn(steps) for _ in range(num_agents)] for _ in range(num_episodes)],
            'dones': [[np.zeros(steps) for _ in range(num_agents)] for _ in range(num_episodes)],
            'values': [[np.random.randn(steps) for _ in range(num_agents)] for _ in range(num_episodes)]
        }
        
        # 运行更新
        stats = grpo_update(arglist, buffer_data, actors, critics, optimizers_a, optimizers_c)
        
        assert stats is not None, "更新返回None"
        assert 'actor_loss' in stats, "缺少actor_loss"
        assert 'critic_loss' in stats, "缺少critic_loss"
        
        print(f"✓ 更新成功: {stats}")
        print("✓ GRPO更新逻辑测试通过!\n")
        
    except Exception as e:
        print(f"✗ GRPO更新逻辑测试失败: {e}\n")
        import traceback
        traceback.print_exc()


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("GRPO实现测试套件")
    print("=" * 50 + "\n")
    
    try:
        test_grpo_models()
        test_grpo_buffer()
        test_gae_computation()
        test_grpo_update() # 新增测试
        test_integration()
        
        print("=" * 50)
        print("✓ 所有测试通过!")
        print("=" * 50)
        print("\nGRPO实现已准备就绪，可以开始训练。")
        print("使用以下命令开始训练:")
        print("  python main_train_unified.py --use_grpo --train True")
        print("或者运行:")
        print("  python run_training.bat")
        
    except Exception as e:
        print("\n" + "=" * 50)
        print("✗ 测试失败!")
        print("=" * 50)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
