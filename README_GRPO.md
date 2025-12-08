# GRPO算法实现说明

## 概述

本项目现在支持两种多智能体强化学习算法：
- **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient) - 基准算法
- **GRPO** (Group Relative Policy Optimization) - 改进算法

## 文件结构

### 新增文件

```
MARL/
├── Model_GRPO.py           # GRPO的Actor和Critic网络定义
├── grpo_buffer.py          # GRPO的经验回放缓冲区
└── train_util_grpo.py      # GRPO训练工具函数

主目录/
├── main_train_grpo.py      # GRPO独立训练脚本
├── main_train_unified.py   # 统一训练入口（支持两种算法）
├── compare_algorithms.py   # 算法对比工具
└── README_GRPO.md          # 本说明文档
```

### 保留的原始文件

```
MARL/
├── Model.py                # MADDPG的Actor和Critic网络
├── replay_buffer.py        # MADDPG的经验回放缓冲区
└── train_util.py           # MADDPG训练工具函数

主目录/
├── main_train.py           # MADDPG训练脚本（保持不变）
└── maddpg.py              # MADDPG原始实现（保持不变）
```

## GRPO算法特点

### 与MADDPG的主要区别

1. **策略优化方式**
   - MADDPG: 使用确定性策略梯度和经验回放
   - GRPO: 使用随机策略梯度和轨迹收集

2. **训练方式**
   - MADDPG: Off-policy，每步更新
   - GRPO: On-policy，收集多个episode后批量更新

3. **优势计算**
   - MADDPG: 使用Q函数直接计算
   - GRPO: 使用GAE（Generalized Advantage Estimation）和组相对优势

4. **关键特性**
   - 使用PPO风格的clip机制防止策略更新过大
   - 组内相对优势归一化，减少组间差异影响
   - 熵正则化鼓励探索

## 评估指标

系统现在支持以下关键性能指标，用于评估和对比算法性能：

1. **订单完成率 (Order Completion Rate)**
   - 定义：成功完成的订单占总分配订单的百分比。
   - 意义：衡量系统处理任务的基本能力。

2. **订单完成时间 (Order Completion Time)**
   - 定义：完成单个订单所需的平均时间。
   - 意义：衡量系统的响应速度和效率。

3. **无人机拦截率 (Interception Rate)**
   - 定义：成功拦截的来袭无人机数量占总来袭无人机数量的比例。
   - 意义：衡量整体防御策略的有效性。

4. **资源利用效率 (Resource Efficiency)**
   - 定义：消耗资源总成本与被摧毁目标的威胁等级总和的比值。
   - 意义：评估算法的效费比（越低越好，或者定义为威胁/成本则越高越好，当前实现为成本/威胁）。

5. **威胁中和率 (Threat Neutralization Rate)**
   - 定义：成功中和的高威胁等级无人机数量占高威胁无人机总数的比例。
   - 意义：衡量系统应对高价值威胁的能力。

这些指标会在训练过程中记录到TensorBoard，并可以通过 `compare_algorithms.py` 进行可视化对比。

## 使用方法

### 1. 训练GRPO模型

#### 方式A：使用统一入口（推荐）

```bash
# 训练GRPO
python main_train_unified.py --use_grpo --train True --num_agents 3 --max_episode 10000

# 训练MADDPG（对比基准）
python main_train_unified.py --train True --num_agents 3 --max_episode 10000
```

#### 方式B：使用独立脚本

```bash
# 训练GRPO
python main_train_grpo.py --train True

# 训练MADDPG
python main_train.py --train True
```

### 2. GRPO关键参数

在 `MARL/arguments.py` 中添加了以下GRPO专用参数：

```python
--grpo_buffer_size      # GRPO缓冲区大小（存储的episode数量），默认10
--grpo_update_interval  # 更新策略的间隔（每N个episode更新一次），默认10
--grpo_epochs          # 每次更新时的训练轮数，默认4
--clip_param           # PPO的clip参数，默认0.2
--entropy_coef         # 熵正则化系数，默认0.01
--gae_lambda           # GAE的lambda参数，默认0.95
--use_grpo            # 是否使用GRPO算法，默认False
```

### 3. 评估模型

```bash
# 评估GRPO模型
python main_train_unified.py --use_grpo --train False --old_model_name models/GRPO_2024-XX-XX-XX-XX-XX/

# 评估MADDPG模型
python main_train_unified.py --train False --old_model_name models/2024-XX-XX-XX-XX-XX/
```

### 4. 对比两种算法

训练完成后，使用对比工具：

```bash
python compare_algorithms.py \
    runs/tensorboard/MADDPG_dispatch_2024-XX-XX-XX-XX-XX \
    runs/tensorboard/GRPO_dispatch_2024-XX-XX-XX-XX-XX \
    comparison_results
```

这将生成：
- `comparison_results/algorithm_comparison.png` - 训练曲线对比图
- `comparison_results/comparison_stats.json` - 详细统计数据

## 训练流程

### GRPO训练流程

```
1. 初始化环境和GRPO智能体（Actor + Critic）
2. 循环训练：
   a. 收集N个完整episode的轨迹
      - 使用当前策略采样动作
      - 存储(obs, action, log_prob, reward, done, value)
   b. 每N个episode后更新策略：
      - 计算GAE优势函数
      - 计算组相对优势
      - 使用PPO-clip更新Actor
      - 使用MSE更新Critic
      - 重复更新K轮
   c. 清空缓冲区，继续收集
3. 保存训练好的模型
```

### MADDPG训练流程（对比）

```
1. 初始化环境和MADDPG智能体（Actor + Critic + Target网络）
2. 循环训练：
   a. 每步收集经验
      - 使用当前策略选择动作
      - 存储到replay buffer
   b. 每步更新策略：
      - 从buffer采样batch
      - 更新Critic（Q函数）
      - 更新Actor（确定性策略）
      - 软更新Target网络
3. 保存训练好的模型
```

## 超参数调优建议

### GRPO推荐配置

```bash
# 基础配置
--num_agents 3
--max_episode 10000
--max_step 200

# GRPO特定配置
--grpo_buffer_size 10          # 适中的buffer size
--grpo_update_interval 10      # 收集10个episode后更新
--grpo_epochs 4                # 每次更新训练4轮
--clip_param 0.2               # 标准PPO clip
--entropy_coef 0.01            # 小的熵系数
--gae_lambda 0.95              # 标准GAE lambda

# 学习率
--lr_a 3e-4                    # Actor学习率
--lr_c 1e-3                    # Critic学习率

# 折扣因子
--gamma 0.95
```

### 调优策略

1. **收敛速度慢**
   - 减小 `grpo_update_interval` (如5)
   - 增加 `grpo_epochs` (如8)
   - 增加 `lr_a` 和 `lr_c`

2. **训练不稳定**
   - 减小 `clip_param` (如0.1)
   - 增加 `grpo_buffer_size` (如20)
   - 减小学习率

3. **探索不足**
   - 增加 `entropy_coef` (如0.05)
   - 减小 `clip_param`

4. **性能不如MADDPG**
   - 调整 `gae_lambda` (尝试0.9-0.99)
   - 增加训练轮数
   - 确保GRPO的网络结构与MADDPG相同

## 实验记录

### 实验日志格式

建议在训练时做好记录：

```
实验日期: 2024-XX-XX
算法: GRPO / MADDPG
参数配置:
  - num_agents: 3
  - grpo_buffer_size: 10
  - grpo_update_interval: 10
  - learning_rate: 3e-4
  ...

结果:
  - 最终平均奖励: XXX
  - 最大奖励: XXX
  - 收敛轮数: XXX
  - 训练时间: XXX分钟
```

## 常见问题

### Q1: 如何选择使用哪个算法？

- **MADDPG**: 
  - 连续动作空间
  - 需要快速响应
  - 样本效率重要
  
- **GRPO**:
  - 离散动作空间
  - 需要更稳定的训练
  - 需要更好的协作性能

### Q2: 两种算法可以用相同的模型吗？

不能直接共用，因为：
- MADDPG使用确定性Actor，输出连续动作
- GRPO使用随机Actor，输出动作概率分布
- GRPO不需要Target网络

### Q3: 如何迁移已有的MADDPG模型到GRPO？

需要重新训练，但可以参考MADDPG的超参数设置。

### Q4: GRPO训练比MADDPG慢吗？

- 单步运行：GRPO稍快（无需每步更新）
- 收敛速度：需实验验证，通常类似或稍慢
- 最终性能：GRPO通常在协作任务中表现更好

## 模型保存格式

### MADDPG模型
```
models/2024-XX-XX-XX-XX-XX/
├── a_a_note.txt
├── a_c_0.pt, a_c_1.pt, a_c_2.pt  # Actor current
├── a_t_0.pt, a_t_1.pt, a_t_2.pt  # Actor target
├── c_c_0.pt, c_c_1.pt, c_c_2.pt  # Critic current
└── c_t_0.pt, c_t_1.pt, c_t_2.pt  # Critic target
```

### GRPO模型
```
models/GRPO_2024-XX-XX-XX-XX-XX/
├── grpo_note.txt
├── grpo_a_0.pt, grpo_a_1.pt, grpo_a_2.pt  # Actor
└── grpo_c_0.pt, grpo_c_1.pt, grpo_c_2.pt  # Critic
```

## 参考资料

1. MADDPG论文: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
2. PPO论文: "Proximal Policy Optimization Algorithms"
3. GAE论文: "High-Dimensional Continuous Control Using Generalized Advantage Estimation"

## 更新日志

- 2024-XX-XX: 初始版本，实现GRPO算法
- 添加统一训练入口
- 添加算法对比工具

## 联系方式

如有问题或建议，请联系开发团队。
