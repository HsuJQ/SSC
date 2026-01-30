# VirtualHome Subgoal Decomposition 技术报告

---

## 1. 本周工作总结

### 1.1 完成的工作

| 序号 | 工作内容 | 状态 |
|------|----------|------|
| 1 | 将SSC方法从Action Sequencing迁移到Subgoal Decomposition任务 | ✅ 完成 |
| 2 | 开发Pangu版本SSC推理脚本 (`client_subgoal_decomposition_self_consistency.py`) | ✅ 完成 |
| 3 | 开发Qwen版本SSC推理脚本 (`qwen_subgoal_decomposition_sc5.py`) | ✅ 完成 |
| 4 | 优化Subgoal Decomposition任务的提示词模板 | ✅ 完成 |
| 5 | 完成Qwen + SSC的推理和评估 | ✅ 完成 |
| 6 | Pangu + SSC推理和评估 | 🔄 进行中 |

### 1.2 产出文件

```
/opt/pangu/examples/vllm-inference/
├── client_subgoal_decomposition_self_consistency.py   # Pangu SSC脚本
├── qwen_subgoal_decomposition_sc5.py                  # Qwen SSC脚本
└── virtualhome_m/
    ├── subgoal_m/
    │   ├── virtualhome_subgoal_decomposition_sc5_outputs.json  # Pangu输出
    │   └── qwen/
    │       └── virtualhome_subgoal_decomposition_qwen_sc5_outputs.json  # Qwen输出
    └── prompts/
        └── optimized_subgoal_prompts.py               # 优化提示词

/opt/pangu/embodied-agent-interface/src/virtualhome_eval/evaluation/subgoal_decomposition/prompts/
└── meta_prompt.py                                     # 优化后的原始模板
```

---

## 2. 方法介绍

### 2.1 结构化自洽推理（SSC）方法

**核心思想**：对同一问题进行多次采样，通过结构化签名和多数投票选出最一致的答案。

**整体流程**：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  输入提示   │ ──▶ │ 多次采样    │ ──▶ │ 结构化签名  │ ──▶ │ 多数投票    │
│  (Prompt)   │     │ (k=5次)     │     │ 生成        │     │ 输出结果    │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

**采样参数**：
- `temperature = 0.7`（增加多样性）
- `top_p = 0.95`
- `max_tokens = 4096`
- `num_samples = 5`

### 2.2 与Action Sequencing SSC的差异（迁移改进点）

#### 任务差异总览

| 对比维度 | Action Sequencing | Subgoal Decomposition | 改进内容 |
|----------|-------------------|----------------------|----------|
| **输出结构** | 扁平字典 `{"FIND": [...]}` | 复合结构 `{"necessity_to_use_action": ..., "output": [...]}` | 多层解析 |
| **签名提取** | 直接取字典key | 正则从字符串提取谓词 | 新算法 |
| **单行内容** | 单个动作 | 可含多谓词（and/or连接） | 全部提取 |
| **有效集合** | 42种动作 | 17种状态谓词 + 19种动作谓词 | 扩展验证 |

---

#### 改进点1：复合结构解析与谓词提取算法

**为什么要改？**

Action Sequencing的输出是扁平的字典结构，动作名称直接作为key：
```json
{"FIND": ["sink", 123], "WALK": ["sink", 123], "GRAB": ["cup", 456]}
```
签名生成非常简单，直接取字典的key拼接即可。

但Subgoal Decomposition的输出是**嵌套的复合结构**：
```json
{
    "necessity_to_use_action": "no",
    "actions_to_include": [],
    "output": ["NEXT_TO(char.65, light.411)", "ON(light.411)"]
}
```
关键信息（子目标序列）藏在`output`数组里，而且每个元素是**字符串**而非结构化数据。必须先定位到`output`字段，再从字符串中**解析出谓词名称**。

**代码对比：**

```python
# Action Sequencing（简单）
def get_signature(data):
    return "->".join(data.keys())  # 直接取key
# {"FIND": [...], "WALK": [...]} → "FIND->WALK"

# Subgoal Decomposition（需要新设计）
def get_signature(data):
    predicates = []
    for subgoal in data.get("output", []):  # 1. 先定位到output数组
        # 2. 用正则从字符串中提取谓词名称
        matches = re.findall(r'([A-Z][A-Z_]+)\s*\(', subgoal)
        predicates.extend(matches)
    return "->".join(predicates)
# {"output": ["NEXT_TO(char.65, light.411)", "ON(light.411)"]} → "NEXT_TO->ON"
```

---

#### 改进点2：多谓词布尔表达式处理

**为什么要改？**

Action Sequencing中，每个动作是独立的key-value对，一次只处理一个动作。

但Subgoal Decomposition中，**一行子目标可能包含多个谓词**，用`and`或`or`连接：
```
"HOLDS_RH(character.65, mouse.413) and HOLDS_LH(character.65, keyboard.415)"
```

如果不处理这种情况，整行会被当作一个单元，导致：
1. 签名不准确（丢失结构信息）
2. 相似的方案因表达顺序不同被判定为不同

**处理方法：**

```python
# 错误做法：整行作为一个单元
subgoal = "HOLDS_RH(char, mouse) and HOLDS_LH(char, keyboard)"
signature = subgoal  # ❌ 无法与其他表达方式匹配

# 正确做法：提取所有谓词
matches = re.findall(r'([A-Z][A-Z_]+)\s*\(', subgoal)
# matches = ["HOLDS_RH", "HOLDS_LH"]  # ✅ 提取出两个谓词
```

这样，以下两种表达会生成相同的签名：
- `"HOLDS_RH(char, mouse) and HOLDS_LH(char, keyboard)"`
- `"HOLDS_LH(char, keyboard) and HOLDS_RH(char, mouse)"`

都提取为：`HOLDS_RH->HOLDS_LH` 或 `HOLDS_LH->HOLDS_RH`（取决于出现顺序）

---

#### 改进点3：一致性验证扩展

**为什么要改？**

Action Sequencing只需验证动作名是否在有效集合中。

但Subgoal Decomposition有**额外的逻辑约束**需要验证：

1. **necessity字段与actions_to_include的一致性**：
   - 若 `necessity_to_use_action = "yes"`，则 `actions_to_include` **不应为空**
   - 若 `necessity_to_use_action = "no"`，则 `actions_to_include` **应为空列表**

2. **谓词类型验证**：
   - 状态谓词（17种）：CLOSED, OPEN, ON, OFF, PLUGGED_IN, SITTING, LYING, ONTOP, INSIDE, NEXT_TO, FACING, HOLDS_RH, HOLDS_LH...
   - 动作谓词（19种）：DRINK, EAT, CUT, LOOKAT, WATCH, GRAB, SWITCHOFF...

**验证代码：**

```python
def validate_subgoal_output(data):
    # 基础结构验证
    if "output" not in data or not isinstance(data["output"], list):
        return False
    
    # 一致性验证（新增）
    necessity = data.get("necessity_to_use_action", "").lower()
    actions = data.get("actions_to_include", [])
    
    if necessity == "yes" and len(actions) == 0:
        return False  # 声称需要动作，但没列出动作
    if necessity == "no" and len(actions) > 0:
        return False  # 声称不需要动作，但列出了动作
    
    return True
```

---

#### 改进点4：提示词优化（针对Subgoal任务特点）

**为什么要改？**

Subgoal Decomposition任务有特定的**前置条件约束**和**时序逻辑要求**，原始提示词没有强调这些，导致模型生成的子目标缺少必要的中间状态。

**常见错误示例：**

```
任务：打开电灯
错误输出：["ON(light.411)"]  # 缺少前置步骤
正确输出：["NEXT_TO(character.65, light.411)", "ON(light.411)"]
```

模型跳过了`NEXT_TO`这个必要的中间状态。

**优化内容：**

在原始提示词基础上，增加以下关键提醒：

```python
# 1. 关键前置条件提醒
- Before operating any object, the character must be NEXT_TO that object first.
  # 为什么：机器人必须先靠近物体才能操作
  
- Before using LOOKAT action, the character must be FACING the target object.
  # 为什么：LOOKAT要求先朝向目标
  
- If an object is INSIDE a closed container, you must OPEN the container first.
  # 为什么：无法直接取出关闭容器内的物品

# 2. 常见操作模式（给模型提供范例）
- To operate a device: NEXT_TO -> FACING -> ON
- To grab an object: NEXT_TO -> HOLDS_RH/HOLDS_LH
- To grab from closed container: NEXT_TO -> OPEN -> HOLDS -> CLOSE

# 3. 输出格式强化
- Output ONLY the JSON object, no explanations, no markdown code blocks.
  # 为什么：模型常在JSON前后加解释文字，导致解析失败
```

**优化效果：**

减少以下错误类型：
- **Runtime Error (MISSING_STEP)**：缺少NEXT_TO等中间状态
- **Grammar Error (Parsing)**：输出格式不符合JSON要求

### 2.3 方法迁移：Action Sequencing → Subgoal Decomposition

| 差异点 | Action Sequencing | Subgoal Decomposition |
|--------|-------------------|----------------------|
| **输出格式** | `{"FIND": [...], "WALK": [...]}` | `{"necessity_to_use_action": ..., "actions_to_include": [...], "output": [...]}` |
| **签名对象** | 动作名称序列 | 谓词名称序列（状态+动作） |
| **验证逻辑** | 验证动作是否在有效集合 | 验证谓词格式 + necessity字段一致性 |
| **有效集合** | 42种动作 | 17种状态谓词 + 19种动作谓词 |

**迁移关键代码**：

```python
# Action Sequencing签名
def get_action_sequence_signature(data):
    return "->".join([action for action in data.keys()])

# Subgoal Decomposition签名（新设计）
def get_subgoal_skeleton_signature(data):
    subgoals = data.get("output", [])
    predicates = []
    for subgoal in subgoals:
        # 提取谓词名称: NEXT_TO(xxx) -> NEXT_TO
        matches = re.findall(r'([A-Z][A-Z_]+)\s*\(', subgoal)
        predicates.extend(matches)
    return "->".join(predicates)
```

---

## 3. 实验对比

### 3.1 实验配置

| 配置项 | 值 |
|--------|-----|
| 数据集 | VirtualHome Subgoal Decomposition |
| 测试样本数 | 338 |
| 采样次数 k | 5 |
| 投票策略 | Skeleton（骨架匹配） |

### 3.2 结果对比

| 模型 | 方法 | 任务成功率 | 成功数/总数 |
|------|------|------------|-------------|
| **Qwen2.5-7B-Instruct** | SSC (k=5) | **22.19%** | 75/338 |
| **Pangu Embedded 7B** | SSC (k=5) | **待补充** | -/338 |
| Qwen2.5-7B-Instruct | Baseline | 待补充 | - |
| Pangu Embedded 7B | Baseline | 待补充 | - |

### 3.3 Qwen + SSC 错误分析

| 错误类型 | 数量 | 占比 | 说明 |
|----------|------|------|------|
| **Runtime** | 30 | 47.6% | 执行时前置条件不满足（如角色在SITTING状态直接FIND） |
| **GoalUnreachable** | 23 | 36.5% | 子目标顺序错误导致目标无法达成 |
| **Hallucination** | 10 | 15.9% | 使用无效谓词（如PUTINSIDE） |

### 3.4 典型错误案例

**Hallucination错误**：
```json
// 错误：使用了不存在的谓词PUTINSIDE
{"output": ["PUTINSIDE(clothes.1003, washing_machine.1001)"]}

// 正确：应使用状态谓词INSIDE
{"output": ["INSIDE(clothes.1003, washing_machine.1001)"]}
```

**Runtime错误**：
```
初始状态: SITTING(character.65)
错误输出: 直接 NEXT_TO(character.65, light.245)
正确输出: 应先 STANDUP(character.65)，再 NEXT_TO
```

---

## 4. 下一步计划

1. 完成Pangu + SSC的推理和评估
2. 完成Baseline对比实验（单次采样）
3. 分析SSC相比Baseline的提升效果
4. 探索提示词进一步优化方向

---

*报告时间: 2026-01-05*
