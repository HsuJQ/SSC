# VirtualHome Goal Interpretation 输出质量改进技术报告

**项目**: Pangu Embedded 7B 模型 - VirtualHome 目标解释任务  
**日期**: 2025年12月13日  
**文档版本**: v1.0

---

## 1. 背景与问题分析

### 1.1 任务描述

VirtualHome Goal Interpretation 任务旨在将自然语言描述的家庭机器人目标转换为符号化目标表示，包括：

- **Node Goals**: 描述物体的期望状态（如 `{"name": "light", "state": "ON"}`）
- **Edge Goals**: 描述物体之间的关系（如 `{"from_name": "character", "relation": "FACING", "to_name": "television"}`）
- **Action Goals**: 描述必须执行的动作（如 `{"action": "WALK", "description": "walk to kitchen"}`）

### 1.2 问题识别

通过对342条模型输出的统计分析，发现以下问题：

| 问题类型 | 数量 | 占比 | 描述 |
|---------|------|------|------|
| 代码块标记 | 240 | 70.2% | 输出包含 ` ```json ` 标记 |
| 键名格式不一致 | 27 | 7.9% | 使用 `node_goals` 而非 `node goals` |
| 包含中文字符 | 40 | 11.7% | 如 `"edge立刻关系"` 等无效键 |
| 推理文本泄漏 | 28 | 8.2% | JSON前有大量推理文字 |
| Action大小写错误 | 7 | 2.0% | 如 `"walk"` 应为 `"WALK"` |

### 1.3 问题示例

```json
// 问题1: 代码块标记
```json
{"node goals": [...]}
```

// 问题2: 键名不一致
{"node_goals": [...], "edge_goals": [...]}

// 问题3: 无效键（含中文）
{"node goals": [...], "edge立刻关系": [...]}

// 问题4: Action大小写
{"action": "walk", "description": "..."}
```

---

## 2. 改进方案设计

### 2.1 整体架构

```
┌───────────────────────────────────────────────────────────────┐
│                 call_vllm_api_with_retry()                    │
│                     (重试机制包装)                             │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    单次尝试流程                          │  │
│  │                                                         │  │
│  │  ┌─────────────────┐                                    │  │
│  │  │   API 调用      │                                    │  │
│  │  │ call_vllm_api() │                                    │  │
│  │  └────────┬────────┘                                    │  │
│  │           │                                             │  │
│  │           ▼                                             │  │
│  │  ┌─────────────────┐                                    │  │
│  │  │  基础后处理     │                                    │  │
│  │  │ parse_output()  │  ← 处理特殊token                   │  │
│  │  └────────┬────────┘                                    │  │
│  │           │                                             │  │
│  │           ▼                                             │  │
│  │  ┌─────────────────────┐                                │  │
│  │  │   增强后处理        │                                │  │
│  │  │ enhanced_parse_     │  ← 移除代码块、统一键名        │  │
│  │  │ output()            │                                │  │
│  │  └────────┬────────────┘                                │  │
│  │           │                                             │  │
│  │           ▼                                             │  │
│  │  ┌─────────────────┐                                    │  │
│  │  │   输出验证      │                                    │  │
│  │  │ validate_output()│  ← 检查JSON结构、必要键           │  │
│  │  └────────┬────────┘                                    │  │
│  │           │                                             │  │
│  │      ┌────┴────┐                                        │  │
│  │      │ 有效?   │                                        │  │
│  │      └────┬────┘                                        │  │
│  │       Yes │ No                                          │  │
│  │           │    │                                        │  │
│  │           │    ▼                                        │  │
│  │           │ ┌─────────────────┐                         │  │
│  │           │ │   JSON修复      │                         │  │
│  │           │ │ clean_and_fix_  │  ← 移除无效键           │  │
│  │           │ │ json()          │                         │  │
│  │           │ └────────┬────────┘                         │  │
│  │           │          │                                  │  │
│  │           │     ┌────┴────┐                             │  │
│  │           │     │修复成功? │                             │  │
│  │           │     └────┬────┘                             │  │
│  │           │      Yes │ No                               │  │
│  │           │          │    │                             │  │
│  │           ▼          ▼    │                             │  │
│  │      ┌──────────────────┐ │                             │  │
│  │      │   返回有效结果   │ │                             │  │
│  │      │  (结束重试循环)  │ │                             │  │
│  │      └──────────────────┘ │                             │  │
│  │                           │                             │  │
│  └───────────────────────────┼─────────────────────────────┘  │
│                              │                                │
│                              ▼                                │
│                    ┌──────────────────┐                       │
│                    │ 重试? (< 3次)    │                       │
│                    └────────┬─────────┘                       │
│                         Yes │ No                              │
│                             │    │                            │
│              ┌──────────────┘    │                            │
│              │ (等待后重新尝试)   │                            │
│              │                   ▼                            │
│              │          ┌──────────────────┐                  │
│              │          │ 返回最后一次结果 │                  │
│              │          │  (标记为无效)    │                  │
│              │          └──────────────────┘                  │
│              │                                                │
└──────────────┴────────────────────────────────────────────────┘
```

### 2.2 模块说明

#### 2.2.1 增强后处理模块 (`enhanced_parse_output`)

**功能**：
1. 移除代码块标记（` ```json `, ` ``` `）
2. 提取JSON部分（跳过推理文本）
3. 统一键名格式（`node_goals` → `node goals`）
4. 修复常见拼写错误（`node_gories` → `node goals`）
5. 修复Action大小写问题

**实现代码**：
```python
def enhanced_parse_output(raw_output):
    if raw_output is None:
        return ""
    
    output = raw_output.strip()
    
    # 1. 移除代码块标记
    output = re.sub(r'^```json\s*', '', output, flags=re.MULTILINE)
    output = re.sub(r'^```\s*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'```$', '', output)
    
    # 2. 提取JSON部分
    json_match = re.search(r'(\{[\s\S]*\})', output)
    if json_match:
        output = json_match.group(1)
    
    # 3. 统一键名格式
    output = re.sub(r'"node_goals"', '"node goals"', output)
    output = re.sub(r'"edge_goals"', '"edge goals"', output)
    output = re.sub(r'"action_goals"', '"action goals"', output)
    
    # 4. 修复Action大小写
    def fix_action_case(match):
        action_name = match.group(1)
        upper_action = action_name.upper()
        if upper_action in VALID_ACTIONS:
            return f'"action": "{upper_action}"'
        return match.group(0)
    
    output = re.sub(r'"action":\s*"([^"]+)"', fix_action_case, output)
    
    return output.strip()
```

#### 2.2.2 输出验证模块 (`validate_output`)

**验证项目**：

| 验证项 | 描述 | 失败处理 |
|-------|------|---------|
| JSON解析 | 检查是否为有效JSON | 返回错误 |
| 必要键检查 | `node goals`, `edge goals`, `action goals` | 返回错误 |
| node goals结构 | 每项需有 `name` 和 `state` | 返回错误 |
| edge goals结构 | 每项需有 `from_name`, `relation`, `to_name` | 返回错误 |
| action goals结构 | 每项需有 `action` 和 `description` | 返回错误 |

**返回值**：
```python
(is_valid: bool, error_msg: str, parsed_data: dict or None)
```

#### 2.2.3 JSON修复模块 (`clean_and_fix_json`)

**修复项目**：
1. 移除包含中文字符的无效键
2. 确保必要键存在（缺失则补充空列表）
3. 过滤无效的relation值
4. 修复action名称大小写

#### 2.2.4 重试机制 (`call_vllm_api_with_retry`)

**参数配置**：
- `max_retries`: 3（最大重试次数）
- `backoff_factor`: 2（退避因子）

**退避策略**：
- 第1次重试：等待 1 秒
- 第2次重试：等待 2 秒
- 第3次重试：等待 4 秒

**流程**：
```
尝试1 → 失败 → 等待1s → 尝试2 → 失败 → 等待2s → 尝试3 → 失败 → 返回最后结果
         ↓              ↓              ↓
       成功           成功           成功
         ↓              ↓              ↓
       返回           返回           返回
```

---

## 3. 配置常量

### 3.1 有效状态集合 (VALID_STATES)

```python
VALID_STATES = {
    'CLOSED', 'OPEN', 'ON', 'OFF', 'SITTING', 'DIRTY', 'CLEAN', 
    'LYING', 'PLUGGED_IN', 'PLUGGED_OUT'
}
```

### 3.2 有效关系集合 (VALID_RELATIONS)

```python
VALID_RELATIONS = {
    'ON', 'INSIDE', 'BETWEEN', 'CLOSE', 'FACING', 'HOLDS_RH', 'HOLDS_LH'
}
```

### 3.3 有效动作集合 (VALID_ACTIONS)

```python
VALID_ACTIONS = {
    'CLOSE', 'DRINK', 'FIND', 'WALK', 'GRAB', 'LOOKAT', 'LOOKAT_SHORT', 
    'LOOKAT_LONG', 'OPEN', 'POINTAT', 'PUTBACK', 'PUTIN', 'PUTOBJBACK', 
    'RUN', 'SIT', 'STANDUP', 'SWITCHOFF', 'SWITCHON', 'TOUCH', 'TURNTO', 
    'WATCH', 'WIPE', 'PUTON', 'PUTOFF', 'GREET', 'DROP', 'READ', 'LIE', 
    'POUR', 'TYPE', 'PUSH', 'PULL', 'MOVE', 'WASH', 'RINSE', 'SCRUB', 
    'SQUEEZE', 'PLUGIN', 'PLUGOUT', 'CUT', 'EAT', 'RELEASE'
}
```

---

## 4. 生成参数优化

### 4.1 参数对比

| 参数 | 优化前 | 优化后 | 说明 |
|------|-------|-------|------|
| `temperature` | 1.0 | 0.3 | 降低随机性，提高格式一致性 |
| `top_p` | 1.0 | 0.9 | 轻度截断低概率token |
| `max_tokens` | 8192 | 8192 | 保持不变 |

### 4.2 参数说明

- **temperature=0.3**: 较低的温度值使模型输出更加确定和一致，减少格式变异
- **top_p=0.9**: 核采样参数，过滤掉概率最低的10%的token，减少异常输出

---

## 5. 统计输出

改进后，每个文件处理完成后会输出统计信息：

```
2025-12-13 10:30:45 - INFO - 文件 virtualhome_goal_interpretation_prompts.json 处理完成:
2025-12-13 10:30:45 - INFO -   总数: 342, 有效: 320, 无效: 22
2025-12-13 10:30:45 - INFO -   重试: 15, 完全失败: 5
2025-12-13 10:30:45 - INFO -   成功率: 93.6%
```

---

## 6. 使用说明

### 6.1 运行命令

```bash
cd /opt/pangu/examples/vllm-inference
python client_generate_0123.py
```

### 6.2 输出格式

输出文件保持原有格式：

```json
[
    {
        "identifier": "27_2",
        "llm_output": "{\"node goals\": [...], \"edge goals\": [...], \"action goals\": [...]}"
    },
    ...
]
```

### 6.3 日志位置

日志文件位于：`/opt/pangu/examples/vllm-inference/client_generate_0123.log`

---

## 7. 预期效果

| 指标 | 改进前 | 预期改进后 |
|------|-------|-----------|
| JSON解析成功率 | ~90% | >98% |
| 键名格式一致性 | ~88% | >99% |
| Action大小写正确率 | ~98% | 100% |
| 无代码块标记 | ~30% | 100% |
| 无中文字符污染 | ~88% | >99% |

---

## 8. 后续优化建议

### 8.1 提示词优化（未实施）

可在提示词末尾添加更强的格式约束：

```
IMPORTANT OUTPUT RULES:
1. Output ONLY valid JSON, no markdown code blocks
2. Use EXACT key names: "node goals", "edge goals", "action goals"
3. All action names must be UPPERCASE
4. Output in English only
```

### 8.2 Few-shot示例（未实施）

在提示词中添加2-3个完整的输入输出示例可以显著提高格式一致性。

### 8.3 严格验证模式（可选）

当前验证模式为宽松模式，可通过取消注释以下代码启用严格验证：

```python
# 状态值验证（严格模式）
if node.get('state') not in VALID_STATES:
    return False, f"Invalid state: {node.get('state')}", None

# 关系值验证（严格模式）
if edge.get('relation') not in VALID_RELATIONS:
    return False, f"Invalid relation: {edge.get('relation')}", None
```

---

## 9. 附录

### 9.1 文件变更清单

| 文件 | 变更类型 | 描述 |
|------|---------|------|
| `client_generate_0123.py` | 修改 | 添加增强后处理、验证、重试机制 |

### 9.2 新增函数清单

| 函数名 | 行数 | 功能 |
|-------|------|------|
| `enhanced_parse_output()` | ~40行 | 增强后处理 |
| `validate_output()` | ~60行 | 输出验证 |
| `clean_and_fix_json()` | ~35行 | JSON修复 |
| `call_vllm_api_with_retry()` | ~50行 | 重试机制 |

### 9.3 依赖项

无新增外部依赖，仅使用Python标准库 `re` 模块（已在文件中导入）。

---

**报告撰写**: GitHub Copilot  
**审核状态**: 待审核
