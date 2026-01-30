# VirtualHome Transition Modeling - SSC方法技术报告

## 1. 本周总结

### 1.1 任务背景

**Transition Modeling（状态转移建模）** 是VirtualHome Embodied Agent评测框架中的核心任务之一。该任务要求模型根据给定的：
- 领域文件（Domain File）：包含谓词定义
- 问题文件（Problem File）：包含对象、初始状态、目标状态
- 未完成的动作模板：包含动作名和参数

编写符合PDDL语法的动作体（precondition和effect），使得从初始状态可以通过执行这些动作达到目标状态。

### 1.2 本周工作

1. **将SSC（Structured Self-Consistency）方法从Action Sequencing/Subgoal Decomposition迁移到Transition Modeling任务**
2. **针对PDDL语法特点优化系统提示词**
3. **完成Pangu和Qwen两个模型的SSC推理脚本适配**
4. **设计针对PDDL结构的签名提取和投票策略**

---

## 2. 方法介绍

### 2.1 SSC方法原理

**SSC (Structured Self-Consistency)** 是一种结构化自洽推理方法，通过多次采样和投票机制提升LLM输出质量：

```
                    ┌─────────────┐
                    │   Prompt    │
                    └─────┬───────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ Sample 1  │   │ Sample 2  │   │ Sample 3  │  ... × k
    │(temp=0.7) │   │(temp=0.7) │   │(temp=0.7) │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │               │
    ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
    │ Extract   │   │ Extract   │   │ Extract   │
    │ Signature │   │ Signature │   │ Signature │
    └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
          │               │               │
          └───────────────┼───────────────┘
                          │
                    ┌─────▼─────┐
                    │   Vote    │
                    │  (选出最  │
                    │  一致方案)│
                    └─────┬─────┘
                          │
                    ┌─────▼─────┐
                    │   Final   │
                    │  Output   │
                    └───────────┘
```

**核心参数配置（与其他任务保持一致）**：
- `num_samples = 5`：每个prompt生成5个样本
- `temperature = 0.7`：较高温度增加多样性
- `top_p = 0.95`：Nucleus采样参数
- `max_tokens = 4096`：最大生成token数

### 2.2 与Action Sequencing/Subgoal Decomposition SSC的差异（迁移改进点）

#### 任务差异总览

| 对比维度 | Action Sequencing | Subgoal Decomposition | Transition Modeling |
|----------|-------------------|----------------------|---------------------|
| **输出内容** | 动作序列 `{"FIND": [...]}` | 子目标序列 `{"output": [...]}` | PDDL动作定义 |
| **输出格式** | 扁平字典 | 嵌套结构 | 复杂PDDL字符串 |
| **签名提取** | 取字典key | 正则提取谓词 | 提取动作名+谓词+结构特征 |
| **有效集合** | 42种动作 | 17+19种谓词 | PDDL关键字+领域谓词 |
| **验证维度** | 动作名有效性 | 谓词+逻辑一致性 | PDDL语法+语义正确性 |

---

#### 改进点1：PDDL结构特征提取算法

**为什么要改？**

Action Sequencing的输出是简单的动作名到参数的映射，签名生成只需拼接key：
```python
# Action Sequencing
signature = "->".join(data.keys())  # "FIND->WALK->GRAB"
```

但Transition Modeling的输出是**复杂的PDDL字符串**，包含嵌套的逻辑结构：
```
(:action walk_towards :parameters (?char - character ?obj - object) 
  :precondition () :effect (next_to ?char ?obj))
(:action switch_on :parameters (?char - character ?obj - object)
  :precondition (and (has_switch ?obj) (next_to ?char ?obj) (off ?obj))
  :effect (and (on ?obj) (not (off ?obj))))
```

必须设计专门的解析算法来提取有意义的结构特征。

**新设计的特征提取：**

```python
def extract_pddl_structure(pddl_output: str) -> Dict:
    """提取PDDL结构特征"""
    return {
        "action_names": extract_action_names(pddl_output),      # ["walk_towards", "switch_on"]
        "predicates_used": extract_predicates_from_pddl(pddl_output),  # ["next_to", "has_switch", "off", "on"]
        "has_or": "(or " in output_lower,                       # 是否使用OR结构
        "has_when": "(when " in output_lower,                   # 是否使用条件效果
        "has_forall": "(forall " in output_lower,               # 是否使用全称量词
        "has_exists": "(exists " in output_lower,               # 是否使用存在量词
        "action_count": len(action_names)                       # 动作数量
    }
```

---

#### 改进点2：多层次签名生成策略

**为什么要改？**

PDDL输出的"正确性"有多个层次：
1. **精确匹配**：完全相同的PDDL字符串
2. **骨架匹配**：动作名和使用的谓词相同，但具体变量名或空格不同
3. **结构匹配**：使用相同的逻辑结构（OR/WHEN/FORALL等）

不同层次适用于不同的投票场景：

```python
# 策略1: 精确匹配（MD5哈希）
def get_full_signature(data):
    normalized = re.sub(r'\s+', ' ', pddl_output.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()

# 策略2: 骨架匹配（动作名+谓词列表）
def get_action_skeleton_signature(data):
    actions = "|".join(sorted(action_names))  # "switch_on|walk_towards"
    predicates = ",".join(sorted(predicates_used))  # "has_switch,next_to,off,on"
    features = []
    if has_or: features.append("OR")
    if has_when: features.append("WHEN")
    return f"{actions}:{predicates}[{','.join(features)}]"

# 策略3: 加权匹配（考虑复杂度）
def get_weighted_signature(data):
    complexity = "simple" if score <= 1 else "moderate" if score <= 3 else "complex"
    return f"{complexity}-{action_count}-{predicate_count}-{skeleton}"
```

---

#### 改进点3：PDDL语法验证

**为什么要改？**

PDDL有严格的语法要求，模型输出可能存在：
- 缺少`:action`关键字
- 括号不匹配
- 使用未定义的谓词

必须增加验证层过滤无效输出：

```python
def validate_transition_output(data: Dict) -> bool:
    """验证transition modeling输出格式"""
    if "output" not in data:
        return False
    
    output_str = data.get("output", "")
    if not isinstance(output_str, str) or not output_str.strip():
        return False
    
    # 必须包含action定义
    if ':action' not in output_str.lower():
        return False
    
    return True
```

---

#### 改进点4：系统提示词优化（针对PDDL任务特点）

**为什么要改？**

PDDL任务有特定的语法规则和领域约束，原始提示词没有强调这些，导致模型生成的动作定义存在常见错误。

**常见错误类型及解决方案：**

| 错误类型 | 示例 | 优化方案 |
|----------|------|----------|
| **缺少前置条件** | switch_on不检查next_to | 添加"Physical Proximity Requirement"提醒 |
| **状态不一致** | on了但没有not (off) | 添加"State Consistency"规则 |
| **属性检查遗漏** | switch_on不检查has_switch | 添加"Object Property Requirements" |
| **语法错误** | 输出非JSON格式 | 强调"Output ONLY JSON, no explanations" |

**优化后的系统提示词关键部分：**

```
## Critical Precondition Rules

### 1. Physical Proximity
To interact with ANY object, character must be next_to that object:
- switch_on/off: (next_to ?char ?obj)
- grab: (next_to ?char ?obj)  

### 2. Object Properties
- switch_on/off requires (has_switch ?obj)
- plug_in/out requires (has_plug ?obj)

### 3. State Consistency
Binary states must be toggled:
- switch_on: precondition (off ?obj), effect (on ?obj) + (not (off ?obj))

## Common Verified Patterns
walk_towards: precondition (), effect (next_to ?char ?obj)
switch_on: precondition (and (has_switch ?obj) (next_to ?char ?obj) (off ?obj))
           effect (and (on ?obj) (not (off ?obj)))
```

---

## 3. 实验对比

### 3.1 实验设置

| 配置项 | Pangu | Qwen |
|--------|-------|------|
| 模型 | pangu_embedded_7b | qwen2.5_7b_instruct |
| API端口 | 1040 | 1043 |
| 停止token | [45892] | [151643] |
| 采样参数 | temp=0.7, top_p=0.95 | temp=0.7, top_p=0.95 |
| 样本数 | 5 | 5 |
| 投票策略 | skeleton | skeleton |

### 3.2 数据集

- 任务：VirtualHome Transition Modeling
- 样本数：300条
- 数据路径：`/opt/pangu/examples/vllm-inference/virtualhome_m/test/generate_prompts/transition_modeling/virtualhome_transition_modeling.json`

### 3.3 实验结果

#### Qwen2.5-7B-Instruct + SSC

| 指标 | 值 |
|------|-----|
| **执行成功率** | 待补充 |
| **任务成功率** | 待补充 |
| 语法错误-解析 | 待补充 |
| 语法错误-幻觉 | 待补充 |
| 运行时错误 | 待补充 |
| 平均置信度 | 待补充 |
| 高置信度比例 (>0.6) | 待补充 |

#### Pangu Embedded 7B + SSC

| 指标 | 值 |
|------|-----|
| **执行成功率** | 待补充 |
| **任务成功率** | 待补充 |
| 语法错误-解析 | 待补充 |
| 语法错误-幻觉 | 待补充 |
| 运行时错误 | 待补充 |
| 平均置信度 | 待补充 |
| 高置信度比例 (>0.6) | 待补充 |

### 3.4 基线对比

| 方法 | Qwen | Pangu |
|------|------|-------|
| Single Sampling | 待补充 | 待补充 |
| SSC (k=5) | 待补充 | 待补充 |
| **提升** | 待补充 | 待补充 |

---

## 4. 运行指南

### 4.1 运行Qwen SSC推理

```bash
cd /opt/pangu/examples/vllm-inference

# 基础运行
python qwen_transition_modeling_sc5.py

# 指定参数运行
python qwen_transition_modeling_sc5.py \
    --num_samples 5 \
    --temperature 0.7 \
    --voting_strategy skeleton \
    --save_meta

# 查看帮助
python qwen_transition_modeling_sc5.py --help
```

### 4.2 运行Pangu SSC推理

```bash
# 基础运行
python client_transition_modeling_self_consistency.py

# 指定参数运行
python client_transition_modeling_self_consistency.py \
    --num_samples 5 \
    --temperature 0.7 \
    --voting_strategy skeleton
```

### 4.3 评估结果

```bash
# 评估Qwen结果
eai-eval \
    --dataset virtualhome \
    --eval-type transition_modeling \
    --mode evaluate_results \
    --llm-response-path /opt/pangu/examples/vllm-inference/virtualhome_m/transition_m/qwen

# 评估Pangu结果
eai-eval \
    --dataset virtualhome \
    --eval-type transition_modeling \
    --mode evaluate_results \
    --llm-response-path /opt/pangu/examples/vllm-inference/virtualhome_m/transition_m
```

---

## 5. 文件结构

```
/opt/pangu/examples/vllm-inference/
├── client_transition_modeling_self_consistency.py    # Pangu SSC脚本
├── qwen_transition_modeling_sc5.py                   # Qwen SSC脚本
└── virtualhome_m/
    └── transition_m/
        ├── output/                                    # Pangu输出
        └── qwen/
            ├── output/                                # Qwen输出
            └── self_consistency_report.md             # 本报告

/opt/pangu/embodied-agent-interface/src/virtualhome_eval/evaluation/transition_modeling/prompts/
└── meta_prompt.py                                     # 优化的系统提示词
```

---

## 6. 总结

本周完成了SSC方法从Action Sequencing/Subgoal Decomposition到Transition Modeling任务的迁移，主要工作包括：

1. **设计PDDL结构特征提取算法**：处理复杂的PDDL输出格式，提取动作名、谓词使用、逻辑结构等特征
2. **实现多层次签名生成策略**：支持精确匹配、骨架匹配、加权匹配三种投票策略
3. **增加PDDL语法验证**：过滤无效的模型输出
4. **优化系统提示词**：添加前置条件规则、状态一致性约束、常见正确模式示例

这些改进使SSC方法能够有效应用于PDDL动作定义任务，通过多次采样和结构化投票提升模型输出质量。
