#!/usr/bin/env python3
"""
VirtualHome Action Sequencing - Self-Consistency 推理脚本

功能：
1. 使用Self-Consistency（自我一致性）方法提高推理质量
2. 对同一个prompt进行多次生成（默认5次）
3. 通过多数投票选出最一致的结果
4. 支持多种投票策略：完全匹配、动作序列匹配、加权投票

用法：
    python client_action_sequencing_self_consistency.py
    python client_action_sequencing_self_consistency.py --num_samples 7 --temperature 0.7
"""

import os
import json
import glob
import requests
import time
import logging
import argparse
import hashlib
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
from tqdm import tqdm
import re

# ==================== 配置部分 ====================
API_URL = "http://127.0.0.1:1040/v1/chat/completions"
MODEL_NAME = "pangu_embedded_7b"

# 路径配置
PROMPT_FILE = "/opt/pangu/examples/vllm-inference/virtualhome/test/generate_prompts/action_sequencing/virtualhome_action_sequencing_prompts.json"
OUTPUT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome/action_m"

# 默认Self-Consistency配置
DEFAULT_NUM_SAMPLES = 5      # 每个prompt生成的样本数
DEFAULT_TEMPERATURE = 0.7    # 较高温度增加多样性
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 4096

# 系统提示词
SYS_PROMPT = "你必须严格遵守法律法规和社会道德规范。" \
    "生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。" \
    "一旦检测到输入或输出有此类倾向，应拒绝回答并发出警告。"

# 有效动作集合
VALID_ACTIONS = {
    'CLOSE', 'DRINK', 'FIND', 'WALK', 'GRAB', 'LOOKAT', 'LOOKAT_SHORT', 
    'LOOKAT_LONG', 'OPEN', 'POINTAT', 'PUTBACK', 'PUTIN', 'PUTOBJBACK', 
    'RUN', 'SIT', 'STANDUP', 'SWITCHOFF', 'SWITCHON', 'TOUCH', 'TURNTO', 
    'WATCH', 'WIPE', 'PUTON', 'PUTOFF', 'GREET', 'DROP', 'READ', 'LIE', 
    'POUR', 'TYPE', 'PUSH', 'PULL', 'MOVE', 'WASH', 'RINSE', 'SCRUB', 
    'SQUEEZE', 'PLUGIN', 'PLUGOUT', 'CUT', 'EAT', 'RELEASE'
}

# ==================== 日志配置 ====================
LOG_FILE = "action_sequencing_self_consistency.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)


# ==================== API 调用 ====================
def call_vllm_api(prompt_text: str, temperature: float = 0.7, 
                   top_p: float = 0.95, max_tokens: int = 4096) -> Optional[str]:
    """
    调用vLLM API进行推理
    
    Args:
        prompt_text: 输入提示
        temperature: 采样温度 (较高增加多样性)
        top_p: nucleus采样参数
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
        "stop_token_ids": [45892],
        "temperature": temperature,
        "top_p": top_p
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
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


# ==================== 输出处理 ====================
def parse_output(output_sent: Optional[str]) -> str:
    """基础后处理：处理特殊token"""
    if output_sent is None:
        return ""
    
    if "[unused17]" in output_sent:
        content = output_sent.split("[unused17]")[-1].split("[unused10]")[0].strip()
    else:
        content = output_sent.strip()
    return content


def extract_json_from_output(raw_output: str) -> Optional[Dict]:
    """
    从原始输出中提取JSON
    
    Args:
        raw_output: 原始模型输出
    
    Returns:
        解析后的字典，失败返回None
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


def normalize_action_sequence(data: Dict) -> Optional[List[Tuple[str, List]]]:
    """
    将action sequencing输出标准化为动作序列
    
    输出格式示例：
    {"FIND": ["sink", 123], "PUTBACK": ["cup", 456, "sink", 123]}
    
    Returns:
        标准化的动作序列列表 [(ACTION, [args]), ...]
    """
    if not data or not isinstance(data, dict):
        return None
    
    action_sequence = []
    for action_name, args in data.items():
        # 标准化动作名称为大写
        normalized_action = action_name.upper()
        
        # 验证是否为有效动作
        if normalized_action not in VALID_ACTIONS:
            continue
        
        # 标准化参数列表
        if args is None:
            args = []
        elif not isinstance(args, list):
            args = [args]
        
        action_sequence.append((normalized_action, args))
    
    return action_sequence if action_sequence else None


def get_action_sequence_signature(data: Dict) -> str:
    """
    获取动作序列的签名（用于投票）
    
    使用动作名称的顺序作为签名，忽略具体参数
    """
    sequence = normalize_action_sequence(data)
    if not sequence:
        return ""
    
    # 只使用动作名称作为签名
    action_names = [action[0] for action in sequence]
    return "->".join(action_names)


def get_full_signature(data: Dict) -> str:
    """
    获取完整签名（包含参数）用于精确匹配
    """
    if not data:
        return ""
    
    # 将字典转换为规范化的JSON字符串
    try:
        normalized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(normalized.encode()).hexdigest()
    except:
        return ""


# ==================== Self-Consistency 核心逻辑 ====================
class SelfConsistencyVoter:
    """Self-Consistency投票器"""
    
    def __init__(self, strategy: str = "action_sequence"):
        """
        Args:
            strategy: 投票策略
                - "exact": 完全匹配（JSON完全相同）
                - "action_sequence": 动作序列匹配（忽略参数顺序）
                - "action_order": 仅比较动作顺序
        """
        self.strategy = strategy
    
    def vote(self, candidates: List[Dict]) -> Tuple[Dict, float, Dict]:
        """
        对候选结果进行投票
        
        Args:
            candidates: 候选的解析结果列表
        
        Returns:
            (最佳结果, 置信度, 投票详情)
        """
        if not candidates:
            return {}, 0.0, {"error": "No candidates"}
        
        if len(candidates) == 1:
            return candidates[0], 1.0, {"single_candidate": True}
        
        # 根据策略生成签名
        signatures = []
        for candidate in candidates:
            if self.strategy == "exact":
                sig = get_full_signature(candidate)
            elif self.strategy == "action_sequence":
                sig = get_action_sequence_signature(candidate)
            else:  # action_order
                sig = get_action_sequence_signature(candidate)
            signatures.append(sig)
        
        # 统计投票
        vote_counter = Counter(signatures)
        
        # 找出最高票的签名
        most_common_sig, vote_count = vote_counter.most_common(1)[0]
        
        # 计算置信度
        confidence = vote_count / len(candidates)
        
        # 找出对应的最佳结果（选择第一个匹配的）
        best_result = None
        for i, sig in enumerate(signatures):
            if sig == most_common_sig:
                best_result = candidates[i]
                break
        
        # 投票详情
        vote_details = {
            "total_candidates": len(candidates),
            "winning_votes": vote_count,
            "confidence": confidence,
            "vote_distribution": dict(vote_counter),
            "strategy": self.strategy
        }
        
        return best_result, confidence, vote_details


def generate_multiple_samples(prompt_text: str, num_samples: int = 5,
                               temperature: float = 0.7, top_p: float = 0.95,
                               max_tokens: int = 4096) -> List[Dict]:
    """
    对同一个prompt生成多个样本
    
    Args:
        prompt_text: 输入提示
        num_samples: 生成样本数
        temperature: 采样温度
        top_p: nucleus采样参数
        max_tokens: 最大token数
    
    Returns:
        解析后的有效结果列表
    """
    candidates = []
    raw_outputs = []
    
    for i in range(num_samples):
        # 调用API
        raw_output = call_vllm_api(prompt_text, temperature, top_p, max_tokens)
        
        if raw_output:
            raw_outputs.append(raw_output)
            
            # 基础后处理
            processed = parse_output(raw_output)
            
            # 提取JSON
            parsed = extract_json_from_output(processed)
            
            if parsed:
                candidates.append(parsed)
    
    logging.debug(f"Generated {len(raw_outputs)} raw outputs, {len(candidates)} valid candidates")
    return candidates


def self_consistency_generate(prompt_text: str, num_samples: int = 5,
                               temperature: float = 0.7, top_p: float = 0.95,
                               max_tokens: int = 4096,
                               voting_strategy: str = "action_sequence") -> Tuple[str, Dict]:
    """
    使用Self-Consistency进行推理
    
    Args:
        prompt_text: 输入提示
        num_samples: 生成样本数
        temperature: 采样温度
        top_p: nucleus采样参数
        max_tokens: 最大token数
        voting_strategy: 投票策略
    
    Returns:
        (最终输出JSON字符串, 元信息字典)
    """
    # 生成多个样本
    candidates = generate_multiple_samples(
        prompt_text, num_samples, temperature, top_p, max_tokens
    )
    
    if not candidates:
        logging.warning("No valid candidates generated")
        return "", {"error": "No valid candidates", "num_samples": num_samples}
    
    # 创建投票器并投票
    voter = SelfConsistencyVoter(strategy=voting_strategy)
    best_result, confidence, vote_details = voter.vote(candidates)
    
    # 构建元信息
    meta_info = {
        "num_samples": num_samples,
        "valid_candidates": len(candidates),
        "confidence": confidence,
        "vote_details": vote_details
    }
    
    # 转换结果为JSON字符串
    try:
        output_str = json.dumps(best_result, ensure_ascii=False)
    except:
        output_str = ""
    
    return output_str, meta_info


# ==================== 主函数 ====================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='VirtualHome Action Sequencing with Self-Consistency'
    )
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f'每个prompt生成的样本数 (默认: {DEFAULT_NUM_SAMPLES})')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'采样温度 (默认: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P,
                        help=f'Top-p采样参数 (默认: {DEFAULT_TOP_P})')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS,
                        help=f'最大生成token数 (默认: {DEFAULT_MAX_TOKENS})')
    parser.add_argument('--voting_strategy', type=str, default='action_sequence',
                        choices=['exact', 'action_sequence', 'action_order'],
                        help='投票策略 (默认: action_sequence)')
    parser.add_argument('--prompt_file', type=str, default=PROMPT_FILE,
                        help=f'提示词文件路径 (默认: {PROMPT_FILE})')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_BASE_PATH,
                        help=f'输出目录 (默认: {OUTPUT_BASE_PATH})')
    parser.add_argument('--save_meta', action='store_true',
                        help='是否保存投票元信息')
    
    args = parser.parse_args()
    
    logging.info("=" * 60)
    logging.info("VirtualHome Action Sequencing - Self-Consistency")
    logging.info("=" * 60)
    logging.info(f"配置:")
    logging.info(f"  样本数: {args.num_samples}")
    logging.info(f"  Temperature: {args.temperature}")
    logging.info(f"  Top-p: {args.top_p}")
    logging.info(f"  投票策略: {args.voting_strategy}")
    logging.info(f"  提示词文件: {args.prompt_file}")
    logging.info(f"  输出目录: {args.output_dir}")
    logging.info("=" * 60)
    
    # 检查输入文件
    if not os.path.exists(args.prompt_file):
        logging.error(f"错误: 找不到提示词文件 {args.prompt_file}")
        return
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取提示词
    try:
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        logging.info(f"读取到 {len(prompts)} 条提示词")
    except Exception as e:
        logging.error(f"读取提示词文件失败: {e}")
        return
    
    # 构建输出文件名
    base_name = os.path.basename(args.prompt_file)
    output_name = base_name.replace("_prompts.json", f"_sc{args.num_samples}_outputs.json")
    if output_name == base_name:
        output_name = base_name.replace(".json", f"_sc{args.num_samples}_outputs.json")
    output_file = os.path.join(args.output_dir, output_name)
    
    # 元信息输出文件
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
    
    # 统计信息
    stats = {
        "total": 0,
        "processed": 0,
        "high_confidence": 0,  # 置信度 > 0.6
        "medium_confidence": 0,  # 置信度 0.4-0.6
        "low_confidence": 0,  # 置信度 < 0.4
        "failed": 0,
        "total_confidence": 0.0
    }
    
    # 处理每个提示词
    for item in tqdm(prompts, desc="Processing", unit="item"):
        identifier = item.get("identifier")
        llm_prompt = item.get("llm_prompt")
        
        if not identifier or not llm_prompt:
            continue
        
        if identifier in processed_ids:
            continue
        
        stats["total"] += 1
        
        # Self-Consistency生成
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
            confidence = meta_info.get("confidence", 0.0)
            stats["total_confidence"] += confidence
            
            if confidence > 0.6:
                stats["high_confidence"] += 1
            elif confidence >= 0.4:
                stats["medium_confidence"] += 1
            else:
                stats["low_confidence"] += 1
            
            # 保存结果
            result_item = {
                "identifier": identifier,
                "llm_output": output_str
            }
            results.append(result_item)
            
            # 保存元信息
            if args.save_meta:
                meta_item = {
                    "identifier": identifier,
                    **meta_info
                }
                meta_results.append(meta_item)
            
            # 每生成一条就保存
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                
                if args.save_meta and meta_results:
                    with open(meta_output_file, 'w', encoding='utf-8') as f:
                        json.dump(meta_results, f, indent=4, ensure_ascii=False)
            except Exception as e:
                logging.error(f"保存文件失败: {e}")
        else:
            stats["failed"] += 1
            logging.warning(f"{identifier} 生成失败")
    
    # 输出统计信息
    logging.info("=" * 60)
    logging.info("处理完成!")
    logging.info("=" * 60)
    logging.info(f"统计信息:")
    logging.info(f"  总数: {stats['total']}")
    logging.info(f"  成功处理: {stats['processed']}")
    logging.info(f"  高置信度 (>0.6): {stats['high_confidence']}")
    logging.info(f"  中置信度 (0.4-0.6): {stats['medium_confidence']}")
    logging.info(f"  低置信度 (<0.4): {stats['low_confidence']}")
    logging.info(f"  失败: {stats['failed']}")
    
    if stats['processed'] > 0:
        avg_confidence = stats['total_confidence'] / stats['processed']
        logging.info(f"  平均置信度: {avg_confidence:.3f}")
        success_rate = (stats['processed'] / stats['total']) * 100
        logging.info(f"  成功率: {success_rate:.1f}%")
    
    logging.info(f"输出文件: {output_file}")
    if args.save_meta:
        logging.info(f"元信息文件: {meta_output_file}")


if __name__ == "__main__":
    main()
