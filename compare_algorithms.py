"""
算法对比工具
用于对比MADDPG和GRPO算法的性能
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def load_tensorboard_data(log_dir, tag='train_episode_rewards'):
    """
    从TensorBoard日志中加载数据
    Args:
        log_dir: TensorBoard日志目录
        tag: 要读取的标签
    Returns:
        steps, values: 步数和对应的值
    """
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        print(f"Warning: Tag '{tag}' not found in {log_dir}")
        return None, None
    
    events = ea.Scalars(tag)
    steps = [event.step for event in events]
    values = [event.value for event in events]
    
    return steps, values


def smooth_curve(values, weight=0.9):
    """
    平滑曲线
    Args:
        values: 原始数据
        weight: 平滑权重
    Returns:
        平滑后的数据
    """
    smoothed = []
    last = values[0]
    for value in values:
        smoothed_val = last * weight + (1 - weight) * value
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def compare_algorithms(maddpg_log_dir, grpo_log_dir, output_dir='comparison_results'):
    """
    对比MADDPG和GRPO算法
    Args:
        maddpg_log_dir: MADDPG的TensorBoard日志目录
        grpo_log_dir: GRPO的TensorBoard日志目录
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    metrics_to_compare = [
        ('train_episode_rewards', 'Average Reward'),
        ('metrics/order_completion_rate', 'Order Completion Rate'),
        ('metrics/avg_order_time', 'Avg Order Completion Time'),
        ('metrics/interception_rate', 'Interception Rate'),
        ('metrics/resource_efficiency', 'Resource Efficiency'),
        ('metrics/threat_neutralization_rate', 'Threat Neutralization Rate')
    ]
    
    all_stats = {}

    for tag, title in metrics_to_compare:
        print(f"处理指标: {title}...")
        
        # 加载数据
        maddpg_steps, maddpg_values = load_tensorboard_data(maddpg_log_dir, tag)
        grpo_steps, grpo_values = load_tensorboard_data(grpo_log_dir, tag)
        
        if maddpg_values is None or grpo_values is None:
            print(f"  无法加载指标 {tag}，跳过")
            continue
        
        # 平滑曲线
        maddpg_smoothed = smooth_curve(maddpg_values, weight=0.9)
        grpo_smoothed = smooth_curve(grpo_values, weight=0.9)
        
        # 绘制对比图
        plt.figure(figsize=(12, 6))
        
        # 原始数据（透明）
        plt.plot(maddpg_steps, maddpg_values, color='blue', alpha=0.2)
        plt.plot(grpo_steps, grpo_values, color='red', alpha=0.2)
        
        # 平滑数据
        plt.plot(maddpg_steps, maddpg_smoothed, color='blue', label='MADDPG', linewidth=2)
        plt.plot(grpo_steps, grpo_smoothed, color='red', label='GRPO', linewidth=2)
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel(title, fontsize=12)
        plt.title(f'MADDPG vs GRPO: {title}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        filename = title.lower().replace(' ', '_') + '.png'
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300)
        plt.close()
        
        # 计算统计信息
        stats = {
            'MADDPG': {
                'final': float(np.mean(maddpg_values[-100:])),
                'max': float(np.max(maddpg_values)),
                'mean': float(np.mean(maddpg_values)),
                'std': float(np.std(maddpg_values))
            },
            'GRPO': {
                'final': float(np.mean(grpo_values[-100:])),
                'max': float(np.max(grpo_values)),
                'mean': float(np.mean(grpo_values)),
                'std': float(np.std(grpo_values))
            }
        }
        all_stats[title] = stats

    # 保存统计信息
    stats_file = os.path.join(output_dir, 'comparison_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=4)
    print(f"统计信息已保存到 {stats_file}")
    
    # 打印统计信息
    print("\n========== 算法对比统计 ==========")
    for title, stats in all_stats.items():
        print(f"\n指标: {title}")
        print(f"  MADDPG 最终: {stats['MADDPG']['final']:.4f}, 平均: {stats['MADDPG']['mean']:.4f}")
        print(f"  GRPO   最终: {stats['GRPO']['final']:.4f}, 平均: {stats['GRPO']['mean']:.4f}")
        
        if abs(stats['MADDPG']['final']) > 1e-6:
            improve = (stats['GRPO']['final'] - stats['MADDPG']['final']) / abs(stats['MADDPG']['final']) * 100
            print(f"  最终改进: {improve:+.2f}%")
    print("=" * 40)


def plot_multiple_runs(log_dirs, labels, output_dir='comparison_results'):
    """
    绘制多次运行的对比图
    Args:
        log_dirs: 日志目录列表
        labels: 标签列表
        output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_dirs)))
    
    for log_dir, label, color in zip(log_dirs, labels, colors):
        steps, rewards = load_tensorboard_data(log_dir, 'train_episode_rewards')
        if rewards is not None:
            smoothed = smooth_curve(rewards, weight=0.9)
            plt.plot(steps, smoothed, label=label, linewidth=2, color=color)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Multi-Run Algorithm Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multi_run_comparison.png'), dpi=300)
    print(f"多次运行对比图已保存到 {output_dir}/multi_run_comparison.png")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 3:
        print("使用方法: python compare_algorithms.py <maddpg_log_dir> <grpo_log_dir> [output_dir]")
        print("\n示例:")
        print("  python compare_algorithms.py runs/tensorboard/MADDPG_dispatch_2024-01-01-12-00-00 runs/tensorboard/GRPO_dispatch_2024-01-01-13-00-00")
        sys.exit(1)
    
    maddpg_log = sys.argv[1]
    grpo_log = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else 'comparison_results'
    
    compare_algorithms(maddpg_log, grpo_log, output)
