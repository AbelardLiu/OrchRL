# 100步长训练配置说明

## 配置概览

### 新配置文件
- **文件**: `orchrl/config/search/search_mas_nosearch_external_roleshare_100step.yaml`
- **启动脚本**: `scripts/run_search_mas_train_100step.sh`

## 主要变更

### 1. 训练步数提升
```yaml
# 之前 (50步)
total_training_steps: 50

# 现在 (100步)
total_training_steps: 100
```

**影响**:
- 训练时间延长约2倍
- 总样本数: ~1,600个 (100步 × 16样本/步)
- 更充分的训练，预期reward提升更明显

### 2. 数据集切换为完整训练集
```yaml
# 之前 (测试采样)
path: .../test_sampled.parquet  # 210个样本

# 现在 (完整训练集)
path: .../train.parquet          # 169,615个样本 (807x更多)
```

**优势**:
- 数据多样性大幅提升
- 避免在小数据集上过拟合
- 模型能学习到更广泛的问题模式

### 3. 验证频率调整
```yaml
# 之前 (50步)
val_freq: 10  # 每10步验证一次 (共5次)

# 现在 (100步)
val_freq: 20  # 每20步验证一次 (共5次)
```

**原因**: 保持验证次数一致，避免过于频繁的checkpoint

### 4. 实验命名更新
```yaml
experiment_name: search_mas_roleshare_100step_train
model_checkpoints_dir: checkpoints/search_mas_roleshare_100step_train
```

## 训练规模对比

| 配置项 | 50步配置 | 100步配置 | 提升 |
|--------|----------|-----------|------|
| **训练步数** | 50 | 100 | 2x |
| **数据集** | test_sampled (210) | train (169,615) | 807x |
| **总样本数** | ~800 | ~1,600 | 2x |
| **验证频率** | 每10步 | 每20步 | - |
| **验证次数** | 5次 | 5次 | - |
| **预计训练时间** | ~30-40分钟 | ~60-80分钟 | 2x |

## 预期效果

### 基于之前的分析

**50步配置 (前10步)**:
- 平均reward: ~1.78%
- 峰值reward: ~7.34%
- 有波动但整体上升趋势

**100步配置预期**:
- 平均reward: 5-10% (更充分的训练)
- 峰值reward: 15-20%
- 更稳定的学习曲线
- 在大数据集上泛化能力更强

## 资源占用

### GPU配置
- **GPU数量**: 2张 (tensor并行)
- **推荐GPU**: GPUs 3,4 (默认)
- **显存/GPU**: ~11-12GB
- **总显存**: ~22-24GB

### 训练时长估算
```
单步时间: ~40-50秒
100步总时长: ~70-90分钟
加上validation: ~80-100分钟 (1.3-1.7小时)
```

## 启动训练

### 使用默认GPU (3,4)
```bash
bash scripts/run_search_mas_train_100step.sh
```

### 指定其他GPU
```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_search_mas_train_100step.sh
```

### 后台运行（推荐长训练）
```bash
nohup bash scripts/run_search_mas_train_100step.sh > train_100step.out 2>&1 &
```

查看进度:
```bash
tail -f logs/search_mas_train_100step_*.log
```

## 监控指标

训练过程中关注以下指标:

1. **mean_reward**: 每步的平均奖励
   - 目标: 从1-2%提升到5-10%

2. **advantages/mean**: 优势函数均值
   - 应该逐渐接近0并保持稳定

3. **critic/score**: Critic评分
   - 应该与实际reward相关

4. **response_length**: 生成长度
   - 太短可能没有完成任务
   - 太长可能超出token限制

## Checkpoint保存

Checkpoints将保存在:
```
checkpoints/search_mas_roleshare_100step_train/
├── global_step_20/
├── global_step_40/
├── global_step_60/
├── global_step_80/
└── global_step_100/
```

最佳模型会被额外保存为 `best_model/`

## 日志位置

- **训练日志**: `logs/search_mas_train_100step_YYYYMMDD_HHMMSS.log`
- **MAS进程日志**: `logs/archives/` (如果启用)
- **输出结果**: `outputs/YYYY-MM-DD/HH-MM-SS/`

## 与之前配置的兼容性

### 技术改进（已应用）
✅ vLLM 0.12.0兼容性修复
✅ GPU显存优化 (gpu_memory_utilization: 0.25)
✅ Partial credit reward (0.0-1.0)
✅ Enhanced prompts
✅ lora_num属性修复

### 架构特性（保持一致）
✅ Role-share模式 (3 agents共享1模型)
✅ 2-GPU tensor并行
✅ Batch size: 8
✅ Sample num: 16

## 故障排查

### 如果训练失败

1. **检查GPU可用性**
   ```bash
   nvidia-smi
   ```

2. **检查数据集路径**
   ```bash
   ls -lh /data1/lll/workspace/multi_agent_rl/DrMAS/mas_app/search/data/drmas_search_mas/train.parquet
   ```

3. **查看错误日志**
   ```bash
   tail -100 logs/search_mas_train_100step_*.log
   ```

4. **显存不足**
   - 降低 `gpu_memory_utilization` 从 0.25 到 0.2
   - 或降低 `train_batch_size` 从 8 到 6

## 下一步

训练完成后:
1. 分析 `TRAINING_RESULTS_ANALYSIS.md` 中的结果
2. 对比50步和100步的reward曲线
3. 评估在测试集上的表现
4. 如果效果好，可以考虑：
   - 增加到200步
   - 使用更大的模型 (1.5B, 3B)
   - 尝试LoRA fine-tuning

---

**创建时间**: 2026-03-11
**状态**: 配置就绪，等待启动
