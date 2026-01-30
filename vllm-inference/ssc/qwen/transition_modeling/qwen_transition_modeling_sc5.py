#!/usr/bin/env python3
"""
VirtualHome Transition Modeling - Qwen Structured Self-Consistency (SSC) 推理脚本

功能：
1. 使用结构化自洽推理(SSC)方法提高PDDL动作定义质量
2. 对同一个prompt进行多次生成（默认5次）
3. 通过多种投票策略选出最一致的结果
4. 完全对齐Pangu版本的参数和方法，确保公平对比

参数设置 (与Pangu保持一致):
- temperature: 0.7
- top_p: 0.95
- max_tokens: 4096
- num_samples: 5

用法：
    python qwen_transition_modeling_sc5.py
    python qwen_transition_modeling_sc5.py --voting_strategy skeleton
"""

import os
import json
import requests
import logging
import argparse
import hashlib
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm

# ==================== 配置部分 ====================
# Qwen API配置
API_URL = "http://127.0.0.1:1043/v1/chat/completions"
MODEL_NAME = "qwen2.5_7b_instruct"
STOP_TOKEN_IDS = [151643]  # Qwen的stop token

# 路径配置
PROMPT_FILE = "/opt/pangu/examples/vllm-inference/virtualhome_m/transition_m/output/virtualhome/generate_prompts/transition_modeling/virtualhome_transition_modeling.json"
OUTPUT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome_m/transition_m/qwen"

# Self-Consistency配置 (与Pangu完全一致)
DEFAULT_NUM_SAMPLES = 5      # 每个prompt生成的样本数
DEFAULT_TEMPERATURE = 0.7    # 与Pangu保持一致
DEFAULT_TOP_P = 0.95         # 与Pangu保持一致
DEFAULT_MAX_TOKENS = 4096    # 与Pangu保持一致

# 系统提示词 - 针对Transition Modeling优化（SSC增强版）
SYS_PROMPT = """You are an expert PDDL planner specializing in VirtualHome domain actions.

Your task is to write the body of PDDL actions given:
1. Domain predicates definitions
2. Problem file with objects, initial conditions, and goals
3. Unfinished action templates with parameters

## Critical Precondition Rules

### 1. Physical Proximity
To interact with ANY object, character must be next_to that object:
- switch_on/off: (next_to ?char ?obj)
- grab: (next_to ?char ?obj)  
- open/close: (next_to ?char ?obj)
- plug_in/out: (next_to ?char ?obj)

### 2. Object Properties
- switch_on/off requires (has_switch ?obj)
- plug_in/out requires (has_plug ?obj)
- grab requires (grabbable ?obj)
- open/close requires (can_open ?obj)

### 3. State Consistency
Binary states must be toggled:
- switch_on: precondition (off ?obj), effect (on ?obj) + (not (off ?obj))
- open: precondition (closed ?obj), effect (open ?obj) + (not (closed ?obj))

## Common Verified Patterns

walk_towards: precondition (), effect (next_to ?char ?obj)

switch_on: precondition (and (has_switch ?obj) (next_to ?char ?obj) (off ?obj))
           effect (and (on ?obj) (not (off ?obj)))

grab: precondition (and (grabbable ?obj) (next_to ?char ?obj))
      effect (or (holds_rh ?char ?obj) (holds_lh ?char ?obj))

open: precondition (and (can_open ?obj) (next_to ?char ?obj) (closed ?obj))
      effect (and (open ?obj) (not (closed ?obj)))

plug_in: precondition (and (has_plug ?obj) (next_to ?char ?obj) (plugged_out ?obj))
         effect (and (plugged_in ?obj) (not (plugged_out ?obj)))

put_on: precondition (or (and (holds_rh ?char ?obj1) (next_to ?char ?obj2)) 
                         (and (holds_lh ?char ?obj1) (next_to ?char ?obj2)))
        effect (and (obj_ontop ?obj1 ?obj2) ...)

standup: precondition (sitting ?char), effect (not (sitting ?char))

## Key Requirements
- Write preconditions in DNF (OR of ANDs)
- ONLY use predicates from domain file exactly
- ONLY use parameters from :parameters section
- Empty () is valid for precondition/effect
- Focus on enabling goal from initial state

## Output Format
{"output": "(:action ... :parameters ... :precondition ... :effect ...) (:action ...)"}

Output ONLY the JSON object, no explanations.
"""

# 有效PDDL谓词集合 (VirtualHome domain)
VALID_PREDICATES = {
    # 状态谓词
    'closed', 'open', 'on', 'off', 'plugged_in', 'plugged_out',
    'sitting', 'lying', 'clean', 'dirty',
    'obj_ontop', 'ontop', 'on_char', 'inside_room', 'obj_inside', 'inside',
    'obj_next_to', 'next_to', 'between', 'facing', 'holds_rh', 'holds_lh',
    # 属性谓词
    'grabbable', 'cuttable', 'can_open', 'readable', 'has_paper',
    'movable', 'pourable', 'cream', 'has_switch', 'lookable', 'has_plug',
    'drinkable', 'body_part', 'recipient', 'containers', 'cover_object',
    'surfaces', 'sittable', 'lieable', 'person', 'hangable', 'clothes', 'eatable'
}

# PDDL关键字
PDDL_KEYWORDS = {
    ':action', ':parameters', ':precondition', ':effect',
    'and', 'or', 'not', 'when', 'forall', 'exists'
}

# ==================== 日志配置 ====================
LOG_FILE = "qwen_transition_modeling_sc5.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


# ==================== API 调用 ====================
def call_qwen_api(prompt_text: str, temperature: float = 0.7, 
                   top_p: float = 0.95, max_tokens: int = 4096) -> Optional[str]:
    """
    调用Qwen API进行推理
    
    Args:
        prompt_text: 输入提示
        temperature: 采样温度 (与Pangu对齐: 0.7)
        top_p: nucleus采样参数 (与Pangu对齐: 0.95)
        max_tokens: 最大生成token数
    
    Returns:
        生成的文本内容，失败返回None
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_tokens,
        "stop_token_ids": STOP_TOKEN_IDS,
        "temperature": temperature,
        "top_p": top_p
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        return content
    except requests.exceptions.Timeout:
        logging.error("API调用超时")
        return None
    except Exception as e:
        logging.error(f"API调用失败: {e}")
        return None


# ==================== 输出处理 (与Pangu版本完全一致) ====================
def parse_output(output_sent: Optional[str]) -> str:
    """基础后处理"""
    if output_sent is None:
        return ""
    return output_sent.strip()


def extract_json_from_output(raw_output: str) -> Optional[Dict]:
    """
    从原始输出中提取JSON (与Pangu版本完全一致)
    """
    if not raw_output:
        return None
    
    output = raw_output.strip()
    
    # 移除代码块标记
    output = re.sub(r'^```json\s*', '', output, flags=re.MULTILINE)
    output = re.sub(r'^```\s*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'```$', '', output)
    output = output.strip()
    
    # 尝试提取JSON部分
    json_match = re.search(r'(\{[\s\S]*\})', output)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # 直接尝试解析
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None


def validate_transition_output(data: Dict) -> bool:
    """
    验证transition modeling输出格式
    """
    if not data or not isinstance(data, dict):
        return False
    
    if "output" not in data:
        return False
    
    output_str = data.get("output", "")
    
    if not isinstance(output_str, str) or not output_str.strip():
        return False
    
    # 基本PDDL结构检查
    output_lower = output_str.lower()
    if ':action' not in output_lower:
        return False
    
    return True


def extract_action_names(pddl_output: str) -> List[str]:
    """
    从PDDL输出中提取动作名称
    """
    pattern = r'\(:action\s+(\w+)'
    matches = re.findall(pattern, pddl_output, re.IGNORECASE)
    return matches


def extract_predicates_from_pddl(pddl_output: str) -> List[str]:
    """
    从PDDL输出中提取使用的谓词名称
    """
    predicates = []
    pattern = r'\((\w+)\s+[?\w]+'
    matches = re.findall(pattern, pddl_output, re.IGNORECASE)
    
    for match in matches:
        match_lower = match.lower()
        if match_lower in VALID_PREDICATES and match_lower not in predicates:
            predicates.append(match_lower)
    
    return predicates


def extract_pddl_structure(pddl_output: str) -> Dict:
    """
    提取PDDL结构特征用于比较
    """
    if not pddl_output:
        return {}
    
    output_lower = pddl_output.lower()
    
    structure = {
        "action_names": extract_action_names(pddl_output),
        "predicates_used": extract_predicates_from_pddl(pddl_output),
        "has_or": " or " in output_lower or "(or " in output_lower,
        "has_when": " when " in output_lower or "(when " in output_lower,
        "has_forall": " forall " in output_lower or "(forall " in output_lower,
        "has_exists": " exists " in output_lower or "(exists " in output_lower,
        "action_count": len(extract_action_names(pddl_output))
    }
    
    return structure


def get_action_skeleton_signature(data: Dict) -> str:
    """
    获取PDDL输出的骨架签名（用于投票）
    """
    if not validate_transition_output(data):
        return ""
    
    pddl_output = data.get("output", "")
    structure = extract_pddl_structure(pddl_output)
    
    if not structure.get("action_names"):
        return ""
    
    # 组合签名：动作名+排序后的谓词列表
    actions = "|".join(sorted(structure["action_names"]))
    predicates = ",".join(sorted(structure["predicates_used"]))
    
    signature = f"{actions}:{predicates}"
    
    # 添加结构特征
    features = []
    if structure["has_or"]:
        features.append("OR")
    if structure["has_when"]:
        features.append("WHEN")
    if structure["has_forall"]:
        features.append("FORALL")
    if structure["has_exists"]:
        features.append("EXISTS")
    
    if features:
        signature += f"[{','.join(features)}]"
    
    return signature


def get_full_signature(data: Dict) -> str:
    """
    获取完整签名用于精确匹配
    """
    if not data:
        return ""
    
    pddl_output = data.get("output", "")
    if not pddl_output:
        return ""
    
    normalized = re.sub(r'\s+', ' ', pddl_output.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def get_weighted_signature(data: Dict) -> str:
    """
    获取加权签名
    """
    if not validate_transition_output(data):
        return ""
    
    pddl_output = data.get("output", "")
    structure = extract_pddl_structure(pddl_output)
    
    action_count = structure.get("action_count", 0)
    predicate_count = len(structure.get("predicates_used", []))
    
    complexity_score = 0
    if structure.get("has_or"):
        complexity_score += 1
    if structure.get("has_when"):
        complexity_score += 1
    if structure.get("has_forall"):
        complexity_score += 2
    if structure.get("has_exists"):
        complexity_score += 2
    
    if complexity_score <= 1:
        complexity = "simple"
    elif complexity_score <= 3:
        complexity = "moderate"
    else:
        complexity = "complex"
    
    skeleton = get_action_skeleton_signature(data)
    
    return f"{complexity}-{action_count}-{predicate_count}-{skeleton[:50]}"


# ==================== Self-Consistency 核心逻辑 ====================
class SelfConsistencyVoter:
    """结构化自洽推理投票器 - Transition Modeling版本"""
    
    def __init__(self, strategy: str = "skeleton"):
        self.strategy = strategy
    
    def vote(self, candidates: List[Dict]) -> Tuple[Dict, float, Dict]:
        if not candidates:
            return {}, 0.0, {"error": "No candidates"}
        
        if len(candidates) == 1:
            return candidates[0], 1.0, {"single_candidate": True}
        
        # 根据策略生成签名
        signatures = []
        for candidate in candidates:
            if self.strategy == "exact":
                sig = get_full_signature(candidate)
            elif self.strategy == "skeleton":
                sig = get_action_skeleton_signature(candidate)
            else:
                sig = get_weighted_signature(candidate)
            signatures.append(sig)
        
        # 过滤空签名
        valid_pairs = [(sig, i) for i, sig in enumerate(signatures) if sig]
        
        if not valid_pairs:
            return candidates[0], 0.2, {"error": "All signatures empty"}
        
        # 统计投票
        vote_counter = Counter([sig for sig, _ in valid_pairs])
        most_common_sig, vote_count = vote_counter.most_common(1)[0]
        
        confidence = vote_count / len(candidates)
        
        best_result = None
        for sig, idx in valid_pairs:
            if sig == most_common_sig:
                best_result = candidates[idx]
                break
        
        vote_details = {
            "total_candidates": len(candidates),
            "valid_candidates": len(valid_pairs),
            "winning_votes": vote_count,
            "confidence": confidence,
            "vote_distribution": dict(vote_counter),
            "strategy": self.strategy,
            "winning_signature": most_common_sig[:80] + "..." if len(most_common_sig) > 80 else most_common_sig
        }
        
        return best_result, confidence, vote_details


def generate_multiple_samples(prompt_text: str, num_samples: int = 5,
                               temperature: float = 0.7, top_p: float = 0.95,
                               max_tokens: int = 4096) -> List[Dict]:
    """
    对同一个prompt生成多个样本
    """
    candidates = []
    raw_outputs = []
    
    for i in range(num_samples):
        raw_output = call_qwen_api(prompt_text, temperature, top_p, max_tokens)
        
        if raw_output:
            raw_outputs.append(raw_output)
            processed = parse_output(raw_output)
            parsed = extract_json_from_output(processed)
            
            if parsed and validate_transition_output(parsed):
                candidates.append(parsed)
    
    logging.debug(f"Generated {len(raw_outputs)} raw outputs, {len(candidates)} valid candidates")
    return candidates


def self_consistency_generate(prompt_text: str, num_samples: int = 5,
                               temperature: float = 0.7, top_p: float = 0.95,
                               max_tokens: int = 4096,
                               voting_strategy: str = "skeleton") -> Tuple[str, Dict]:
    """
    使用Structured Self-Consistency进行推理
    """
    candidates = generate_multiple_samples(
        prompt_text, num_samples, temperature, top_p, max_tokens
    )
    
    if not candidates:
        logging.warning("No valid candidates generated")
        return "", {"error": "No valid candidates", "num_samples": num_samples}
    
    voter = SelfConsistencyVoter(strategy=voting_strategy)
    best_result, confidence, vote_details = voter.vote(candidates)
    
    meta_info = {
        "num_samples": num_samples,
        "valid_candidates": len(candidates),
        "confidence": confidence,
        "vote_details": vote_details
    }
    
    try:
        output_str = json.dumps(best_result, ensure_ascii=False)
    except:
        output_str = ""
    
    return output_str, meta_info


# ==================== 主函数 ====================
def main():
    parser = argparse.ArgumentParser(
        description='VirtualHome Transition Modeling with Qwen SSC'
    )
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f'每个prompt生成的样本数 (默认: {DEFAULT_NUM_SAMPLES})')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'采样温度 (默认: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P,
                        help=f'Top-p采样参数 (默认: {DEFAULT_TOP_P})')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help=f'最大生成token数 (默认: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--voting_strategy', type=str, default='skeleton',
                        choices=['exact', 'skeleton', 'weighted'],
                        help='投票策略 (默认: skeleton)')
    parser.add_argument('--prompt_file', type=str, default=PROMPT_FILE,
                        help=f'提示词文件路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_BASE_PATH,
                        help=f'输出目录')
    parser.add_argument('--save_meta', action='store_true',
                        help='是否保存投票元信息')
    
    args = parser.parse_args()
    
    logging.info("=" * 70)
    logging.info("VirtualHome Transition Modeling - Qwen SSC")
    logging.info("=" * 70)
    logging.info(f"配置:")
    logging.info(f"  模型: {MODEL_NAME}")
    logging.info(f"  样本数: {args.num_samples}")
    logging.info(f"  Temperature: {args.temperature}")
    logging.info(f"  Top-p: {args.top_p}")
    logging.info(f"  投票策略: {args.voting_strategy}")
    logging.info(f"  提示词文件: {args.prompt_file}")
    logging.info(f"  输出目录: {args.output_dir}")
    logging.info("=" * 70)
    
    if not os.path.exists(args.prompt_file):
        logging.error(f"错误: 找不到提示词文件 {args.prompt_file}")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        logging.info(f"读取到 {len(prompts)} 条提示词")
    except Exception as e:
        logging.error(f"读取提示词文件失败: {e}")
        return
    
    base_name = os.path.basename(args.prompt_file)
    output_name = base_name.replace(".json", f"_qwen_sc{args.num_samples}_outputs.json")
    output_file = os.path.join(args.output_dir, output_name)
    
    meta_output_file = output_file.replace("_outputs.json", "_meta.json") if args.save_meta else None
    
    logging.info(f"输出文件: {output_file}")
    
    # 检查断点续传
    existing_results = []
    existing_meta = []
    processed_ids = set()
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                for item in existing_results:
                    processed_ids.add(item.get("identifier"))
            logging.info(f"发现已有结果文件，包含 {len(existing_results)} 条记录，将跳过已完成项。")
        except json.JSONDecodeError:
            logging.warning("已有结果文件损坏，将重新开始。")
    
    if meta_output_file and os.path.exists(meta_output_file):
        try:
            with open(meta_output_file, 'r', encoding='utf-8') as f:
                existing_meta = json.load(f)
        except:
            pass
    
    results = existing_results
    meta_results = existing_meta
    
    stats = {
        "total": 0,
        "processed": 0,
        "high_confidence": 0,
        "medium_confidence": 0,
        "low_confidence": 0,
        "failed": 0,
        "total_confidence": 0.0
    }
    
    for item in tqdm(prompts, desc="Processing Transition Modeling (Qwen)", unit="item"):
        identifier = item.get("identifier")
        llm_prompt = item.get("llm_prompt")
        
        if not identifier or not llm_prompt:
            continue
        
        if identifier in processed_ids:
            continue
        
        stats["total"] += 1
        
        output_str, meta_info = self_consistency_generate(
            llm_prompt,
            num_samples=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            voting_strategy=args.voting_strategy
        )
        
        if output_str:
            stats["processed"] += 1
            confidence = meta_info.get("confidence", 0)
            stats["total_confidence"] += confidence
            
            if confidence > 0.6:
                stats["high_confidence"] += 1
            elif confidence >= 0.4:
                stats["medium_confidence"] += 1
            else:
                stats["low_confidence"] += 1
            
            result_item = {
                "identifier": identifier,
                "output": output_str
            }
            results.append(result_item)
            
            if args.save_meta:
                meta_item = {
                    "identifier": identifier,
                    "meta": meta_info
                }
                meta_results.append(meta_item)
            
            if len(results) % 10 == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                if meta_output_file:
                    with open(meta_output_file, 'w', encoding='utf-8') as f:
                        json.dump(meta_results, f, ensure_ascii=False, indent=2)
        else:
            stats["failed"] += 1
            logging.warning(f"Failed to generate output for {identifier}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if meta_output_file:
        with open(meta_output_file, 'w', encoding='utf-8') as f:
            json.dump(meta_results, f, ensure_ascii=False, indent=2)
    
    logging.info("=" * 70)
    logging.info("处理完成!")
    logging.info("=" * 70)
    logging.info(f"统计信息:")
    logging.info(f"  总数: {stats['total']}")
    logging.info(f"  成功: {stats['processed']}")
    logging.info(f"  失败: {stats['failed']}")
    if stats['processed'] > 0:
        avg_confidence = stats['total_confidence'] / stats['processed']
        logging.info(f"  平均置信度: {avg_confidence:.3f}")
        logging.info(f"  高置信度 (>0.6): {stats['high_confidence']}")
        logging.info(f"  中置信度 (0.4-0.6): {stats['medium_confidence']}")
        logging.info(f"  低置信度 (<0.4): {stats['low_confidence']}")
    logging.info(f"  输出文件: {output_file}")
    if meta_output_file:
        logging.info(f"  元信息文件: {meta_output_file}")


if __name__ == "__main__":
    main()
