#!/usr/bin/env python3
"""
训练结果对比工具
用于对比原始训练和改进后训练的效果
"""

import re
import sys
from pathlib import Path


def extract_metrics(log_file):
    """从训练日志中提取关键指标"""
    if not Path(log_file).exists():
        return None

    with open(log_file, 'r') as f:
        content = f.read()

    # 提取每个step的reward
    rewards_pattern = r'step:(\d+).*?verifier_model_critic/rewards/mean:([\d.]+).*?searcher_model_critic/rewards/mean:([\d.]+).*?answerer_model_critic/rewards/mean:([\d.]+)'

    steps = []
    for match in re.finditer(rewards_pattern, content, re.DOTALL):
        step = int(match.group(1))
        verifier_reward = float(match.group(2))
        searcher_reward = float(match.group(3))
        answerer_reward = float(match.group(4))
        steps.append({
            'step': step,
            'verifier_reward': verifier_reward,
            'searcher_reward': searcher_reward,
            'answerer_reward': answerer_reward,
            'avg_reward': (verifier_reward + searcher_reward + answerer_reward) / 3
        })

    # 提取advantages
    adv_pattern = r'step:(\d+).*?verifier_model_critic/advantages/mean:([-\d.]+).*?searcher_model_critic/advantages/mean:([-\d.]+).*?answerer_model_critic/advantages/mean:([-\d.]+)'

    advantages = {}
    for match in re.finditer(adv_pattern, content, re.DOTALL):
        step = int(match.group(1))
        advantages[step] = {
            'verifier_adv': float(match.group(2)),
            'searcher_adv': float(match.group(3)),
            'answerer_adv': float(match.group(4))
        }

    return {'steps': steps, 'advantages': advantages}


def print_comparison(baseline_log, improved_log):
    """打印对比结果"""
    baseline = extract_metrics(baseline_log) if baseline_log else None
    improved = extract_metrics(improved_log) if improved_log else None

    print("=" * 80)
    print("📊 训练结果对比")
    print("=" * 80)

    if baseline and baseline['steps']:
        print("\n🔵 Baseline训练 (5步, 4样本/步):")
        print("-" * 80)
        print(f"{'Step':<6} {'Verifier':>10} {'Searcher':>10} {'Answerer':>10} {'Average':>10}")
        print("-" * 80)
        for s in baseline['steps']:
            print(f"{s['step']:<6} {s['verifier_reward']:>10.3f} {s['searcher_reward']:>10.3f} "
                  f"{s['answerer_reward']:>10.3f} {s['avg_reward']:>10.3f}")

        if baseline['steps']:
            last_step = baseline['steps'][-1]
            print("-" * 80)
            print(f"最终平均Reward: {last_step['avg_reward']:.3f}")

            if last_step['step'] in baseline['advantages']:
                adv = baseline['advantages'][last_step['step']]
                print(f"最终Advantages: Verifier={adv['verifier_adv']:.3f}, "
                      f"Searcher={adv['searcher_adv']:.3f}, Answerer={adv['answerer_adv']:.3f}")
    else:
        print("\n🔵 Baseline训练: 日志未找到或未完成")

    if improved and improved['steps']:
        print("\n\n🟢 改进后训练 (50步, 16样本/步, Partial Credit):")
        print("-" * 80)
        print(f"{'Step':<6} {'Verifier':>10} {'Searcher':>10} {'Answerer':>10} {'Average':>10}")
        print("-" * 80)

        # 显示前10步、中间几步、最后几步
        steps_to_show = []
        if len(improved['steps']) <= 15:
            steps_to_show = improved['steps']
        else:
            steps_to_show = improved['steps'][:5] + \
                          [{'step': '...', 'verifier_reward': 0, 'searcher_reward': 0,
                            'answerer_reward': 0, 'avg_reward': 0}] + \
                          improved['steps'][-5:]

        for s in steps_to_show:
            if s['step'] == '...':
                print(f"{'...':^6} {'...':>10} {'...':>10} {'...':>10} {'...':>10}")
            else:
                print(f"{s['step']:<6} {s['verifier_reward']:>10.3f} {s['searcher_reward']:>10.3f} "
                      f"{s['answerer_reward']:>10.3f} {s['avg_reward']:>10.3f}")

        if improved['steps']:
            last_step = improved['steps'][-1]
            print("-" * 80)
            print(f"最终平均Reward: {last_step['avg_reward']:.3f}")

            if last_step['step'] in improved['advantages']:
                adv = improved['advantages'][last_step['step']]
                print(f"最终Advantages: Verifier={adv['verifier_adv']:.3f}, "
                      f"Searcher={adv['searcher_adv']:.3f}, Answerer={adv['answerer_adv']:.3f}")

            # 计算改进幅度
            if baseline and baseline['steps']:
                baseline_final = baseline['steps'][-1]['avg_reward']
                improved_final = last_step['avg_reward']
                improvement = ((improved_final - baseline_final) / (baseline_final + 1e-6)) * 100
                print(f"\n📈 改进幅度: {improvement:+.1f}%")
                print(f"   Baseline最终: {baseline_final:.3f}")
                print(f"   Improved最终: {improved_final:.3f}")
    else:
        print("\n\n🟢 改进后训练: 尚未开始或进行中...")
        print("   请运行: bash scripts/run_search_mas_train_improved.sh")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    baseline_log = "logs/search_mas_train_e2e_20260311_111214.log"  # 最近的baseline日志
    improved_log = None

    # 查找最新的improved日志
    logs_dir = Path("logs")
    if logs_dir.exists():
        improved_logs = sorted(logs_dir.glob("search_mas_train_improved_*.log"))
        if improved_logs:
            improved_log = str(improved_logs[-1])

    if len(sys.argv) > 1:
        baseline_log = sys.argv[1]
    if len(sys.argv) > 2:
        improved_log = sys.argv[2]

    print_comparison(baseline_log, improved_log)
