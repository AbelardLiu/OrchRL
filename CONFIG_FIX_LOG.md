# 训练配置修复日志

## 问题1: GPU显存不足 (第一次失败)

**错误**: `CUDA out of memory. Tried to allocate 5.81 GiB. GPU has 4.01 GiB free.`

**原因**:
- batch_size=8, sample_num=16 对24GB GPU太大
- 显存占用: 19.66 GiB / 23.68 GiB

**修复**:
- batch_size: 8 → 6
- sample_num: 16 → 12
- token_length: 8192 → 6144
- 添加 micro_batch_size = 2

## 问题2: Token长度参数不一致 (第二次失败)

**错误**: `AssertionError: max_token_len must be greater than the sequence length. Got max_token_len=6144 and max_seq_len=8192`

**原因**:
- ppo_max_token_len_per_gpu = 6144 ✓
- 但 max_prompt_length + max_response_length = 4096 + 4096 = 8192 ✗
- 导致 8192 > 6144 冲突

**修复**: 对齐所有token length参数
```yaml
training:
  max_prompt_length: 3072
  max_response_length: 3072

models:
  model_*:
    ppo_trainer_config:
      data:
        max_prompt_length: 3072
        max_response_length: 3072
      actor_rollout_ref:
        actor:
          ppo_max_token_len_per_gpu: 6144
        rollout:
          prompt_length: 3072
          response_length: 3072
          max_num_batched_tokens: 6144
```

**验证**: 3072 + 3072 = 6144 ≤ 6144 ✓

## 最终配置 (search_mas_nosearch_external_50step_memory_optimized.yaml)

### 训练优化
- total_training_steps: 50 (vs 5, 10x提升)
- train_batch_size: 6 (vs 4, 1.5x提升)
- train_sample_num: 12 (vs 4, 3x提升)
- Total samples: 600 (vs 20, 30x提升)

### 显存优化
- max_prompt_length: 3072 (vs 4096)
- max_response_length: 3072 (vs 4096)
- ppo_max_token_len_per_gpu: 6144 (vs 8192)
- ppo_micro_batch_size_per_gpu: 2 (新增)
- log_prob_micro_batch_size_per_gpu: 2 (vs 4)
- max_num_seqs: 96 (vs 64)
- 预期显存: ~17-18 GiB (vs 20+ GiB)

### Reward & Prompt优化
- Partial credit reward (0.0-1.0分层)
- Enhanced prompts (减少幻觉)

## 预期效果

虽然略微降低了token长度，但对search任务足够：
- 平均问题长度: ~50-200 tokens
- 平均检索信息: ~500-1000 tokens
- 平均回答长度: ~100-300 tokens
- 总计: ~650-1500 tokens (远小于6144)

**预期最终reward**: 40-60% (vs baseline 6-20%)

## 运行命令

```bash
bash scripts/run_search_mas_train_improved.sh
```

## 日期

- 2026-03-11: 初次配置优化
- 2026-03-11: 修复GPU OOM
- 2026-03-11: 修复token长度不一致

## 问题3: GPU显存仍然不足 - 采用Role-Share模式 (2026-03-11)

### 问题
即使降低到batch_size=6, sample_num=12, token_length=6144，显存仍然不足：
- 尝试分配: 5.95 GiB
- 可用显存: 2.87 GiB
- 总显存占用: 20.80 GiB / 23.68 GiB

**根本原因**: 3个独立模型占用太多显存（每个~7GB）

### 解决方案: Role-Share模式

**核心思想**:
- 3个agent共享同一个LLM模型
- 使用2-GPU tensor并行训练
- 大幅降低显存占用

**架构变化**:
```
原始 (Specialization):        Role-Share:
GPU 0: Verifier Model      GPU 0: Shared Model (part 1)
GPU 1: Searcher Model  =>  GPU 1: Shared Model (part 2)
GPU 2: Answerer Model      (unused)
Total: ~21GB               Total: ~16GB (-25%)
```

### 配置文件: search_mas_nosearch_external_roleshare_50step.yaml

```yaml
specialization: shared  # 关键：从full改为shared

base_models:
  shared_policy:  # 单一共享模型
    path: /data1/lll/models/Qwen3-0.6B
    name: shared_model

agent_policy_configs:
  agent_configs:
    agent_0:
      policy_name: shared_model  # 3个agent都指向同一个模型
    agent_1:
      policy_name: shared_model
    agent_2:
      policy_name: shared_model

training:
  mate:
    role_policy_mapping:
      verifier: shared_model   # 所有role映射到同一个模型
      searcher: shared_model
      answerer: shared_model

models:
  shared_model:  # 只有一个模型配置
    ppo_trainer_config:
      actor_rollout_ref:
        rollout:
          tensor_model_parallel_size: 2  # 2-GPU模型并行
        trainer:
          n_gpus_per_node: 2  # 只使用2张GPU
```

### 参数优化

| 参数 | Baseline | 3-Model | Role-Share |
|------|----------|---------|------------|
| Training steps | 5 | 50 | **50** ✓ |
| Samples/step | 4 | 12 | **16** ✓ |
| Batch size | 4 | 6 | **8** ✓ |
| 模型数量 | 3 | 3 | **1** ✓ |
| GPU使用 | 3 | 3 | **2** ✓ |
| Tensor并行 | 1 | 1 | **2** ✓ |
| 显存/GPU | ~7GB | ~20GB | **~8GB** ✓ |

### 优势

1. **显存大幅降低**
   - 3个模型 → 1个共享模型: 节省66%参数显存
   - 2-GPU tensor并行: 每GPU负担减半
   - 总显存: ~21GB → ~16GB

2. **训练效率提升**
   - 更大batch size: 4 → 8
   - 更多samples: 4 → 16
   - 单一模型训练更稳定

3. **可扩展性好**
   - 添加新agent不需要新模型
   - 统一策略网络
   - 更容易部署

### 技术细节

**Tensor并行**:
- 模型参数分布在2张GPU上
- GPU 0: 模型前半部分层
- GPU 1: 模型后半部分层
- 自动同步梯度和参数

**角色区分**:
- 通过不同的prompts区分角色
- Enhanced prompts确保角色明确
- Credit assignment支持多角色学习

### 预期效果

- 显存占用: GPU 0/1 各 ~8-10 GiB
- 总训练样本: 800 (50步 × 16样本, vs baseline 20)
- 预期最终reward: 50-70% (vs baseline 6-20%)

### 运行命令

```bash
bash scripts/run_search_mas_train_improved.sh
```

或指定GPU:
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_search_mas_train_improved.sh
```

## 问题4: vLLM Import错误 (2026-03-11)

### 错误
```
ImportError: cannot import name 'current_stream' from 'vllm.utils'
```

### 原因
- vLLM 0.12.0移除了`vllm.utils.current_stream`
- monkey patch代码尝试导入不存在的函数

### 解决方案
修改`verl/verl/workers/rollout/vllm_rollout/monkey_patch.py:84-85`:
```python
# Before:
from vllm.utils import current_stream
stream = current_stream()

# After:
# vLLM 0.12.0+: use torch.cuda.current_stream() directly
stream = torch.cuda.current_stream()
```

## 问题5: GPU显存分配冲突 (2026-03-11)

### 错误
```
ValueError: Free memory on device (6.26/23.68 GiB) on startup is less than desired GPU memory utilization (0.6, 14.21 GiB).
```

### 原因
- FSDP训练模型加载占用约17-18GB显存
- vLLM推理引擎在同样的GPU上初始化，默认`gpu_memory_utilization=0.6`需要14.21GB
- 实际只剩余5-6GB可用显存

### 解决方案
在role-share配置中降低vLLM的GPU内存利用率：
```yaml
models:
  shared_model:
    ppo_trainer_config:
      actor_rollout_ref:
        rollout:
          gpu_memory_utilization: 0.25  # 从0.6降到0.25
```

这样vLLM只需要约6GB显存（23.68 * 0.25 = 5.92GB），能在剩余显存中运行。

### 预期显存占用
- FSDP训练模型：约17-18GB（分布在2个GPU）
- vLLM推理引擎：约6GB（分布在2个GPU，使用tensor并行）
- 总计：每GPU约11-12GB

## 问题6: 缺少lora_num属性导致checkpoint保存失败 (2026-03-11)

### 错误
```
AttributeError: 'AsyncActorRolloutRefWorker' object has no attribute 'lora_num'
```

### 原因
- `lora_num`只在使用LoRA训练时才被设置（仅当`self._is_actor and self._is_lora`时）
- 但在`save_checkpoint`方法中，调试代码无条件访问`self.lora_num`
- Role-share配置不使用LoRA（`lora_rank: 0`），因此该属性未初始化

### 解决方案
修改`verl/verl/workers/fsdp_workers.py:1092`:
```python
# Before:
print(f"[rank-{self.rank}]: lora_num={self.lora_num}")

# After:
print(f"[rank-{self.rank}]: lora_num={getattr(self, 'lora_num', 1)}")
```

### 训练进度
训练成功启动并运行，完成了约24次采样迭代后在第一次validation checkpoint时失败。修复后可以继续训练。

## 问题7: 配置100步长训练 (2026-03-11)

### 需求
1. 增加训练步数: 50 → 100
2. 切换到完整训练集: test_sampled.parquet → train.parquet

### 数据集对比
```
test_sampled.parquet: 210个样本
train.parquet:        169,615个样本 (807x更多)
```

### 新配置文件
创建 `search_mas_nosearch_external_roleshare_100step.yaml`:

**主要变更**:
```yaml
# 训练步数
total_training_steps: 100  # 从50增加

# 数据集
prompt_loader:
  path: .../train.parquet  # 从test_sampled切换到train

# 验证频率
val_freq: 20  # 从10调整，保持验证次数一致

# 实验名称
experiment_name: search_mas_roleshare_100step_train
model_checkpoints_dir: checkpoints/search_mas_roleshare_100step_train
```

### 训练规模

| 指标 | 50步配置 | 100步配置 | 提升 |
|------|----------|-----------|------|
| 训练步数 | 50 | 100 | 2x |
| 数据集大小 | 210 | 169,615 | 807x |
| 总样本数 | ~800 | ~1,600 | 2x |
| 预计时长 | 30-40分钟 | 80-100分钟 | 2x |

### 启动脚本
创建 `scripts/run_search_mas_train_100step.sh`:
```bash
bash scripts/run_search_mas_train_100step.sh
```

### 预期效果
- 平均reward: 5-10% (vs 50步的1.78%)
- 峰值reward: 15-20% (vs 50步的7.34%)
- 在大数据集上泛化能力更强
- 学习曲线更稳定

详细文档: `TRAINING_100STEP_CONFIG.md`
