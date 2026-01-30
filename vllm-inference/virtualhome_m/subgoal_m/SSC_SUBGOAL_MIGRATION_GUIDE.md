# VirtualHome Subgoal Decomposition - SSC 方法迁移指南

## 概述

本文档介绍如何将 **Structured Self-Consistency (SSC)** 方法从 Action Sequencing 任务迁移到 Subgoal Decomposition 任务。

## 文件清单

### 新创建的脚本

| 文件 | 描述 | 模型 |
|------|------|------|
| `client_subgoal_decomposition_self_consistency.py` | Pangu基础SSC脚本 | Pangu Embedded 7B |
| `qwen_subgoal_decomposition_sc5.py` | Qwen对齐版SSC脚本 | Qwen2.5-7B-Instruct |
| `client_subgoal_decomposition_ssc_enhanced.py` | 增强版SSC脚本（支持优化提示词） | Pangu Embedded 7B |

### 提示词文件

| 文件 | 描述 |
|------|------|
| `virtualhome_m/prompts/optimized_subgoal_prompts.py` | 优化版提示词模板 |

## 任务差异对比

### Action Sequencing
```
输入: 任务描述 + 场景物体
输出: {"FIND": ["sink", 123], "WALK": ["sink", 123], ...}
签名: ACTION1->ACTION2->ACTION3
```

### Subgoal Decomposition
```
输入: 初始状态 + 目标状态 + 场景物体
输出: {
    "necessity_to_use_action": "yes/no",
    "actions_to_include": ["ACTION1", ...],
    "output": ["NEXT_TO(char, obj)", "ON(device)", ...]
}
签名: PREDICATE1->PREDICATE2->PREDICATE3
```

## 参数配置

所有脚本使用统一的参数配置，确保公平对比：

| 参数 | 值 | 说明 |
|------|-----|------|
| `temperature` | 0.7 | 采样温度（增加多样性） |
| `top_p` | 0.95 | Nucleus采样参数 |
| `max_tokens` | 4096 | 最大生成token数 |
| `num_samples` | 5 | 每个prompt的采样次数 |

## 投票策略

### 1. Exact (精确匹配)
- 使用MD5哈希对完整JSON进行签名
- 要求输出完全一致
- 最严格，置信度可能较低

### 2. Skeleton (骨架匹配) - **推荐**
- 只提取谓词名称序列作为签名
- 忽略具体参数（物体ID等）
- 适合Subgoal任务的特点

### 3. Weighted (加权投票)
- 结合子目标数量和骨架特征
- 优先选择步骤数适中(4-8步)的方案

## 使用方法

### Pangu模型推理
```bash
# 基础版本
python client_subgoal_decomposition_self_consistency.py \
    --num_samples 5 \
    --voting_strategy skeleton \
    --save_meta

# 增强版本（使用优化提示词）
python client_subgoal_decomposition_ssc_enhanced.py \
    --use_optimized_prompt \
    --voting_strategy skeleton \
    --save_meta
```

### Qwen模型推理
```bash
python qwen_subgoal_decomposition_sc5.py \
    --num_samples 5 \
    --voting_strategy skeleton \
    --save_meta
```

## 输出文件

### 结果文件
```
virtualhome_m/subgoal_m/
├── virtualhome_subgoal_decomposition_sc5_outputs.json       # Pangu结果
├── virtualhome_subgoal_decomposition_sc5_meta.json          # Pangu元信息
├── qwen/
│   ├── virtualhome_subgoal_decomposition_qwen_sc5_outputs.json  # Qwen结果
│   └── virtualhome_subgoal_decomposition_qwen_sc5_meta.json     # Qwen元信息
```

### 输出格式
```json
[
    {
        "identifier": "scene_1_27_2",
        "llm_output": "{\"necessity_to_use_action\": \"no\", \"actions_to_include\": [], \"output\": [\"NEXT_TO(character.65, washing_machine.1001)\", \"ON(washing_machine.1001)\"]}"
    }
]
```

## 签名生成示例

对于子目标输出:
```json
{
    "output": [
        "NEXT_TO(character.65, computer.417)",
        "FACING(character.65, computer.417)",
        "ON(computer.417)"
    ]
}
```

生成的骨架签名:
```
NEXT_TO->FACING->ON
```

## 提示词优化

优化后的系统提示词包含：

1. **更清晰的任务描述**
   - 明确角色定位（机器人任务规划器）
   - 简化谓词词汇表

2. **关键规则强调**
   - 时序逻辑（不同行有先后顺序）
   - 前置条件检查（FACING before LOOKAT）
   - 中间状态添加

3. **常见模式提示**
   - 操作设备: NEXT_TO -> FACING -> [operation]
   - 抓取物体: NEXT_TO -> [OPEN if needed] -> HOLDS

4. **严格输出格式**
   - 只输出JSON，不要解释

## 评估

生成结果后，使用评估脚本：
```bash
cd /opt/pangu/embodied-agent-interface
python -m src.virtualhome_eval.evaluation.subgoal_decomposition.evaluate \
    --input_file <output_file> \
    --output_dir <eval_output_dir>
```

## 注意事项

1. **API端口**
   - Pangu: 1040
   - Qwen: 1043

2. **断点续传**
   - 脚本支持自动续传，中断后重新运行会跳过已完成项

3. **元信息**
   - 使用 `--save_meta` 保存投票详情，用于分析

4. **日志**
   - 日志文件：`subgoal_decomposition_self_consistency.log`
   - 包含详细的处理进度和错误信息

## 创新点总结

1. **任务适配**: 将SSC方法从动作序列生成适配到子目标分解
2. **谓词骨架签名**: 设计新的签名方法，提取状态/动作谓词序列
3. **加权投票优化**: 考虑子目标数量的合理性（4-8步为优）
4. **提示词增强**: 针对子目标任务特点优化系统提示词
