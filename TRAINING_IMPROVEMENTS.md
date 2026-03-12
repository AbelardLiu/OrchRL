# 训练优化改进总结

## 📊 问题诊断

### 原始配置的问题
1. ❌ 训练步数太少（仅5步）→ 前4步完全0% reward
2. ❌ 样本量太小（4 samples/step）→ 统计不可靠
3. ❌ Binary reward（0/1）→ 学习信号稀疏
4. ❌ Advantages为负 → Critic未学到有效策略
5. ❌ 模型产生幻觉答案 → Prompt不够严格

### 训练结果
```
Step 1-4: 0% success rate (完全失败)
Step 5:   6-20% success rate
  - Verifier:  10% (2/20)
  - Searcher:  20% (1/5)
  - Answerer:  6.25% (1/16)

Advantages: 全部为负或零（-0.042, 0.000, -0.001）
```

## ✅ 改进措施

### 1. 训练配置优化 (orchrl/config/search/search_mas_nosearch_external_50step_improved.yaml)

| 参数 | 原值 | 新值 | 改进倍数 |
|------|------|------|----------|
| total_training_steps | 5 | **50** | 10x |
| train_sample_num | 4 | **16** | 4x |
| train_batch_size | 4 | **8** | 2x |
| val_freq | 9999 | **10** | 定期验证 |
| log_prob_micro_batch_size | 4 | **8** | 2x |
| max_num_seqs | 64 | **128** | 2x |

**预期效果**：
- 更多训练步数让模型充分学习
- 更大样本量提高统计可靠性
- 定期验证监控训练进度

### 2. Reward函数改进 (orchrl/reward/search/external_mas_reward.py)

**原函数**: Binary reward (0.0 或 1.0)
```python
final_reward = 1.0 if _is_correct(predicted, expected_candidates) else 0.0
```

**新函数**: Partial credit reward (0.0 - 1.0)
```python
def _compute_answer_score(predicted: str, expected_candidates: list[str]) -> float:
    """
    Reward分层:
    - 1.0: 完全匹配
    - 0.8: 答案完全包含在预测中（>50% overlap）
    - 0.6: 答案包含在预测中
    - 0.5: 预测大部分包含在答案中（>50% overlap）
    - 0.4: 词汇重叠 >50%
    - 0.3: 预测部分包含在答案中
    - 0.2: 词汇重叠 >30%
    - 0.0: 完全不匹配
    """
```

**预期效果**：
- 提供更密集的学习信号
- 部分正确的回答也能获得正向反馈
- 帮助模型渐进式改进

### 3. Prompt优化 (examples/mas_app/search/search_mas/apps/search/prompts.py)

#### Verifier Agent Prompt
- ✅ 强调基于检索信息验证
- ✅ 明确只有DIRECT evidence才能verify yes
- ✅ 避免基于general knowledge判断

#### Search Agent Prompt
- ✅ 强调生成SPECIFIC查询
- ✅ 提示关注关键事实（名字、日期、地点）
- ✅ 避免过于宽泛的查询

#### Answer Agent Prompt（最重要）
- ✅ **强制要求只能基于检索信息回答**
- ✅ **明确禁止编造或猜测**
- ✅ **信息不足时返回"I don't have enough information"**
- ✅ 提供清晰的示例

**预期效果**：
- 大幅减少幻觉答案（如"Nigeria's Marcusonnen"）
- 提高答案质量和可信度
- 更好地利用检索信息

### 4. 新训练脚本 (scripts/run_search_mas_train_improved.sh)

- ✅ 使用新配置文件
- ✅ 清晰显示改进参数
- ✅ 保持所有必要的环境变量

## 🚀 如何使用

### 方法1: 使用新脚本（推荐）
```bash
bash scripts/run_search_mas_train_improved.sh
```

### 方法2: 手动指定配置
```bash
export CONFIG_NAME=search_mas_nosearch_external_50step_improved
bash scripts/run_search_mas_train_e2e.sh
```

## 📈 预期改进效果

### 短期（前10步）
- Mean reward从0%提升到10-30%
- Advantages从负数变为正数
- 幻觉答案显著减少

### 中期（10-30步）
- Mean reward提升到30-50%
- Critic网络稳定，advantages稳定为正
- 模型开始学到可复现策略

### 长期（30-50步）
- Mean reward达到50-70%+
- 三个agent协同工作更好
- Answerer学会基于检索信息准确回答

## 📝 监控指标

训练时重点关注：
1. **Mean reward趋势**: 应该持续上升
2. **Advantages**: 应该从负变正并稳定
3. **Response quality**: 检查archived logs中的答案质量
4. **Success rate**: 每10步验证一次

## 🔧 进一步优化建议

如果50步后效果仍不理想：
1. 增加到100-200步训练
2. 调整learning rate
3. 尝试curriculum learning（从简单问题开始）
4. 增加retrieval service质量
5. 考虑使用更大的base model (0.6B → 1.8B)

## 📂 新增文件

1. `orchrl/config/search/search_mas_nosearch_external_50step_improved.yaml` - 改进的配置
2. `scripts/run_search_mas_train_improved.sh` - 新训练脚本
3. `TRAINING_IMPROVEMENTS.md` - 本文档

## 🔄 修改的文件

1. `orchrl/reward/search/external_mas_reward.py` - Partial credit reward
2. `examples/mas_app/search/search_mas/apps/search/prompts.py` - 优化的prompts

---

## ⚠️ GPU显存优化 (2026-03-11 更新)

### 问题
初次运行改进配置时遇到 **CUDA Out of Memory** 错误：
- 尝试分配: 5.81 GiB
- 可用显存: 4.01 GiB  
- 总显存: 23.68 GiB (已使用 19.66 GiB)

### 解决方案
创建显存优化配置: `search_mas_nosearch_external_50step_memory_optimized.yaml`

### 优化措施对比

| 参数 | 原始值 | 过大配置 | 显存优化 |
|------|--------|----------|----------|
| total_training_steps | 5 | 50 | **50** ✓ |
| train_batch_size | 4 | 8 | **6** ✓ |
| train_sample_num | 4 | 16 | **12** ✓ |
| ppo_max_token_len_per_gpu | 8192 | 8192 | **6144** ✓ |
| ppo_micro_batch_size_per_gpu | - | - | **2** ✓ |
| log_prob_micro_batch_size | 4 | 8 | **2** ✓ |
| max_num_batched_tokens | 8192 | 8192 | **6144** ✓ |
| max_num_seqs | 64 | 128 | **96** ✓ |

### 关键优化
1. **Micro batch size = 2**: 使用梯度累积，减少单次前向传播显存占用
2. **Token length: 8192 → 6144**: 降低25%的序列长度，大幅降低显存峰值
3. **Batch size: 8 → 6**: 平衡训练效率和显存占用
4. **Sample num: 16 → 12**: 仍保持3倍提升

### 训练效果保证
虽然略微降低参数，但相比baseline仍有巨大提升：
- Training steps: 5 → **50** (10x)
- Total samples: 20 → **600** (30x)
- Partial credit reward ✓
- Enhanced prompts ✓

**预期最终reward**: 40-60% (vs baseline 6-20%)

### 使用方法
训练脚本已自动更新，直接运行：
```bash
bash scripts/run_search_mas_train_improved.sh
```

### 如果仍然OOM
1. 设置环境变量: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
2. 进一步减小batch/sample: `train_batch_size: 4`, `train_sample_num: 8`
3. 减小max_prompt/response_length: 设为 3072
