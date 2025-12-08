# GRPO算法实现 - 快速开始指南

## 🎯 项目改动概述

本项目已成功添加GRPO（Group Relative Policy Optimization）算法支持，现在您可以：
1. ✅ 使用原有的MADDPG算法作为基准
2. ✅ 使用新的GRPO算法进行训练
3. ✅ 轻松对比两种算法的性能

**所有原有代码保持不变，可随时切换回MADDPG！**

---

## 📁 新增文件清单

### 核心算法实现
```
MARL/
├── Model_GRPO.py           # GRPO的Actor和Critic网络
├── grpo_buffer.py          # GRPO专用经验缓冲区
└── train_util_grpo.py      # GRPO训练工具函数
```

### 训练和评估脚本
```
主目录/
├── main_train_grpo.py      # GRPO独立训练脚本
├── main_train_unified.py   # 统一训练入口（推荐使用）
├── test_grpo.py           # GRPO实现测试脚本
└── run_training.bat       # 一键训练脚本（Windows）
```

### 对比和文档
```
主目录/
├── compare_algorithms.py   # 算法性能对比工具
├── README_GRPO.md         # 详细使用说明
└── QUICKSTART.md          # 本文件
```

---

## 🚀 快速开始

### 方式1: 使用一键脚本（最简单）

```bash
# Windows用户
run_training.bat

# 然后根据提示选择：
#   [1] 训练MADDPG
#   [2] 训练GRPO
#   [3] 评估MADDPG
#   [4] 评估GRPO
#   [5] 对比算法
```

### 方式2: 命令行训练

#### 训练GRPO（新算法）
```bash
python main_train_unified.py --use_grpo --train True --num_agents 3 --max_episode 10000
```

#### 训练MADDPG（基准算法）
```bash
python main_train_unified.py --train True --num_agents 3 --max_episode 10000
```

#### 评估GRPO模型
```bash
python main_train_unified.py --use_grpo --train False --old_model_name models/GRPO_2024-XX-XX-XX-XX-XX/
```

#### 评估MADDPG模型
```bash
python main_train_unified.py --train False --old_model_name models/2024-XX-XX-XX-XX-XX/
```

---

## 📊 对比两种算法

训练完成后，使用对比工具：

```bash
python compare_algorithms.py \
    runs/tensorboard/MADDPG_dispatch_2024-XX-XX-XX-XX-XX \
    runs/tensorboard/GRPO_dispatch_2024-XX-XX-XX-XX-XX \
    comparison_results
```

生成的文件：
- `comparison_results/algorithm_comparison.png` - 训练曲线对比图
- `comparison_results/comparison_stats.json` - 统计数据

---

## ⚙️ GRPO关键参数

在`MARL/arguments.py`中新增的参数：

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--use_grpo` | False | 是否使用GRPO算法 |
| `--grpo_buffer_size` | 10 | 缓冲区大小（episode数） |
| `--grpo_update_interval` | 10 | 更新间隔（每N个episode） |
| `--grpo_epochs` | 4 | 每次更新的训练轮数 |
| `--clip_param` | 0.2 | PPO clip参数 |
| `--entropy_coef` | 0.01 | 熵正则化系数 |
| `--gae_lambda` | 0.95 | GAE lambda参数 |
| `--no_record_metrics` | False | 训练时不记录详细metrics文件（节省空间） |

---

## 🔍 算法对比

### MADDPG特点
- ✅ Off-policy学习，样本效率高
- ✅ 适合连续动作空间
- ✅ 训练稳定
- ❌ 需要较大的replay buffer
- ❌ 在某些协作任务中表现一般

### GRPO特点
- ✅ On-policy学习，更新稳定
- ✅ 适合离散动作空间
- ✅ 组相对优势，提升协作性能
- ✅ PPO-style clip，防止策略崩溃
- ❌ 需要收集完整轨迹
- ❌ 可能需要更多训练轮数

---

## 📝 建议的对比实验流程

### 第一步：训练基准模型（MADDPG）
```bash
python main_train_unified.py --train True --num_agents 3 --max_episode 10000 --note "MADDPG基准实验"
```

### 第二步：训练GRPO模型
```bash
python main_train_unified.py --use_grpo --train True --num_agents 3 --max_episode 10000 --note "GRPO对比实验"
```

### 第三步：对比性能
```bash
# 找到两个训练生成的日志目录
# 位于 runs/tensorboard/ 下
python compare_algorithms.py <MADDPG日志路径> <GRPO日志路径>
```

### 第四步：评估最佳模型
```bash
# 评估100个episode
python main_train_unified.py --train False --eval_episode 100 --old_model_name <最佳模型路径>
```

---

## 🔧 调试和测试

### 测试GRPO实现
```bash
python test_grpo.py
```

这将测试：
- ✓ GRPO模型结构
- ✓ 缓冲区功能
- ✓ GAE计算
- ✓ 与环境集成

### 查看TensorBoard
```bash
tensorboard --logdir runs/tensorboard
```
然后在浏览器访问 `http://localhost:6006`

---

## 📚 详细文档

更多细节请参考：
- `README_GRPO.md` - 完整的使用说明和技术细节
- `MARL/arguments.py` - 所有可配置参数
- `main_train_unified.py` - 统一训练入口源码

---

## ❓ 常见问题

### Q: 我的旧代码还能用吗？
**A:** 完全可以！所有原有文件（`main_train.py`, `maddpg.py`等）保持不变。

### Q: 如何只用MADDPG？
**A:** 使用原有脚本或统一入口不加`--use_grpo`参数即可。

### Q: GRPO训练更慢吗？
**A:** 单步执行更快（无需每步更新），但需要收集完整轨迹。总体时间相近。

### Q: 如何选择算法？
**A:** 
- 连续动作 → MADDPG
- 离散动作 + 需要协作 → GRPO
- 不确定 → 两个都试试，用对比工具选最好的

### Q: 训练不收敛怎么办？
**A:** 
1. 检查超参数（学习率、clip_param等）
2. 增加训练轮数
3. 调整buffer_size和update_interval
4. 参考README_GRPO.md的调优建议

---

## 🎓 推荐学习路径

1. **第一天**：运行`test_grpo.py`确保环境正常
2. **第二天**：用少量episode（如1000）快速训练两种算法
3. **第三天**：对比结果，调整超参数
4. **第四天**：用完整配置（10000+ episodes）训练最优模型
5. **第五天**：详细评估和分析

---

## 📞 获取帮助

如遇问题：
1. 查看 `README_GRPO.md` 详细文档
2. 运行 `test_grpo.py` 检查实现
3. 检查TensorBoard日志
4. 联系开发团队

---

## ✨ 开始探索

现在您可以：
```bash
# 1. 测试实现
python test_grpo.py

# 2. 快速训练对比
python main_train_unified.py --use_grpo --train True --max_episode 1000
python main_train_unified.py --train True --max_episode 1000

# 3. 查看结果
tensorboard --logdir runs/tensorboard
```

**祝实验顺利！🚀**
