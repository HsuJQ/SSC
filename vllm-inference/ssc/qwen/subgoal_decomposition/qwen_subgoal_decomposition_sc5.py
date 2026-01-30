#!/usr/bin/env python3
"""
VirtualHome Subgoal Decomposition - Qwen Structured Self-Consistency (SSC) 推理脚本

功能：
1. 使用结构化自洽推理(SSC)方法提高子目标分解质量
2. 对同一个prompt进行多次生成（默认5次）
3. 通过多种投票策略选出最一致的结果
4. 完全对齐Pangu版本的参数和方法，确保公平对比

参数设置 (与Pangu保持一致):
- temperature: 0.7
- top_p: 0.95
- max_tokens: 4096
"""

import os
import json
import logging
import argparse
from tqdm import tqdm
from ssc.qwen.subgoal_decomposition.constants import ALL_VALID_PREDICATES
from ssc.qwen.subgoal_decomposition.runner import self_consistency_generate

API_URL = "http://127.0.0.1:1043/v1/chat/completions"
MODEL_NAME = "qwen2.5_7b_instruct"
STOP_TOKEN_IDS = [151643]

PROMPT_FILE = "/opt/pangu/examples/vllm-inference/virtualhome_m/test/generate_prompts/subgoal_decomposition/virtualhome_subgoal_decomposition.json"
OUTPUT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome_m/subgoal_m/qwen"

DEFAULT_NUM_SAMPLES = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 4096

LOG_FILE = "qwen_subgoal_decomposition_sc5.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# 导入系统提示词
import sys
sys.path.insert(0, '/opt/pangu/embodied-agent-interface/src/virtualhome_eval/evaluation/subgoal_decomposition/prompts')
from meta_prompt import system_prompt as TASK_SYSTEM_PROMPT
SYS_PROMPT = TASK_SYSTEM_PROMPT


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


def validate_subgoal_output(data: Dict) -> bool:
    """
    验证subgoal decomposition输出格式
    """
    if not data or not isinstance(data, dict):
        return False
    
    if "output" not in data:
        return False
    
    if not isinstance(data.get("output"), list):
        return False
    
    return True


def normalize_subgoal_sequence(data: Dict) -> Optional[List[str]]:
    """
    将subgoal输出标准化为子目标序列
    """
    if not validate_subgoal_output(data):
        return None
    
    output_list = data.get("output", [])
    if not output_list:
        return None
    
    normalized_subgoals = []
    for subgoal in output_list:
        if isinstance(subgoal, str):
            cleaned = subgoal.strip()
            if cleaned:
                normalized_subgoals.append(cleaned)
    
    return normalized_subgoals if normalized_subgoals else None


def extract_predicates_from_subgoal(subgoal: str) -> List[str]:
    """
    从子目标表达式中提取谓词名称
    """
    predicates = []
    pattern = r'([A-Z][A-Z_]+)\s*\('
    matches = re.findall(pattern, subgoal)
    
    for match in matches:
        if match in ALL_VALID_PREDICATES:
            predicates.append(match)
    
    return predicates


def get_subgoal_skeleton_signature(data: Dict) -> str:
    """
    获取子目标序列的骨架签名（用于投票）
    """
    subgoals = normalize_subgoal_sequence(data)
    if not subgoals:
        return ""
    
    all_predicates = []
    for subgoal in subgoals:
        preds = extract_predicates_from_subgoal(subgoal)
        all_predicates.extend(preds)
    
    if not all_predicates:
        return ""
    
    return "->".join(all_predicates)


def get_full_signature(data: Dict) -> str:
    """
    获取完整签名（MD5）用于精确匹配
    """
    if not data:
        return ""
    
    try:
        normalized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(normalized.encode()).hexdigest()
    except:
        return ""


def get_weighted_signature(data: Dict) -> str:
    """
    获取加权签名：结合子目标数量和骨架特征
    """
    subgoals = normalize_subgoal_sequence(data)
    if not subgoals:
        return ""
    
    skeleton = get_subgoal_skeleton_signature(data)
    length = len(subgoals)
    
    if 4 <= length <= 8:
        weight_prefix = "optimal"
    elif length < 4:
        weight_prefix = "short"
    else:
        weight_prefix = "long"
    
    return f"{weight_prefix}-{length}-{skeleton}"


# ==================== Self-Consistency 核心逻辑 ====================
class SelfConsistencyVoter:
    """结构化自洽推理投票器 - 与Pangu版本完全一致"""
    
    def __init__(self, strategy: str = "skeleton"):
        """
        Args:
            strategy: 投票策略
                - "exact": 完全匹配（JSON完全相同，MD5签名）
                - "skeleton": 骨架匹配（只比较谓词序列）
                - "weighted": 加权投票（考虑子目标数量和骨架）
        """
        self.strategy = strategy
    
    def vote(self, candidates: List[Dict]) -> Tuple[Dict, float, Dict]:
        """
        对候选结果进行投票
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
            elif self.strategy == "skeleton":
                sig = get_subgoal_skeleton_signature(candidate)
            else:  # weighted
                sig = get_weighted_signature(candidate)
            signatures.append(sig)
        
        # 过滤空签名
        valid_pairs = [(sig, i) for i, sig in enumerate(signatures) if sig]
        
        if not valid_pairs:
            return candidates[0], 0.2, {"error": "All signatures empty"}
        
        # 统计投票
        vote_counter = Counter([sig for sig, _ in valid_pairs])
        
        # 找出最高票的签名
        most_common_sig, vote_count = vote_counter.most_common(1)[0]
        
        # 计算置信度
        confidence = vote_count / len(candidates)
        
        # 找出对应的最佳结果
        best_result = None
        for sig, idx in valid_pairs:
            if sig == most_common_sig:
                best_result = candidates[idx]
                break
        
        # 投票详情
        vote_details = {
            "total_candidates": len(candidates),
            "valid_candidates": len(valid_pairs),
            "winning_votes": vote_count,
            "confidence": confidence,
            "vote_distribution": dict(vote_counter),
            "strategy": self.strategy,
            "winning_signature": most_common_sig[:50] + "..." if len(most_common_sig) > 50 else most_common_sig
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
            
            if parsed and validate_subgoal_output(parsed):
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
        description='VirtualHome Subgoal Decomposition - Qwen with SSC (Aligned with Pangu)'
    )
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES,
                        help=f'每个prompt生成的样本数 (默认: {DEFAULT_NUM_SAMPLES})')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                        help=f'采样温度 (默认: {DEFAULT_TEMPERATURE}, 与Pangu对齐)')
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P,
                        help=f'Top-p采样参数 (默认: {DEFAULT_TOP_P}, 与Pangu对齐)')
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
    logging.info("Qwen Subgoal Decomposition - Structured Self-Consistency (SSC)")
    logging.info("=" * 70)
    logging.info(f"配置 (与Pangu对齐):")
    logging.info(f"  模型: {MODEL_NAME}")
    logging.info(f"  API: {API_URL}")
    logging.info(f"  样本数: {args.num_samples}")
    logging.info(f"  Temperature: {args.temperature}")
    logging.info(f"  Top-p: {args.top_p}")
    logging.info(f"  投票策略: {args.voting_strategy}")
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
    
    # 构建输出文件名
    output_name = f"virtualhome_subgoal_decomposition_qwen_sc{args.num_samples}_outputs.json"
    output_file = os.path.join(args.output_dir, output_name)
    meta_output_file = output_file.replace("_outputs.json", "_meta.json") if args.save_meta else None
    
    logging.info(f"输出文件: {output_file}")
    
    # 断点续传
    existing_results = []
    existing_meta = []
    processed_ids = set()
    
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
                for item in existing_results:
                    processed_ids.add(item.get("identifier"))
            logging.info(f"发现已有结果文件，包含 {len(existing_results)} 条记录")
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
    
    for item in tqdm(prompts, desc="Processing (Qwen SSC)", unit="item"):
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
            confidence = meta_info.get("confidence", 0.0)
            stats["total_confidence"] += confidence
            
            if confidence > 0.6:
                stats["high_confidence"] += 1
            elif confidence >= 0.4:
                stats["medium_confidence"] += 1
            else:
                stats["low_confidence"] += 1
            
            result_item = {
                "identifier": identifier,
                "llm_output": output_str
            }
            results.append(result_item)
            
            if args.save_meta:
                meta_item = {"identifier": identifier, **meta_info}
                meta_results.append(meta_item)
            
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
    
    logging.info("=" * 70)
    logging.info("处理完成!")
    logging.info("=" * 70)
    logging.info(f"统计信息:")
    logging.info(f"  总数: {stats['total']}")
    logging.info(f"  成功: {stats['processed']}")
    logging.info(f"  高置信度: {stats['high_confidence']}")
    logging.info(f"  中置信度: {stats['medium_confidence']}")
    logging.info(f"  低置信度: {stats['low_confidence']}")
    logging.info(f"  失败: {stats['failed']}")
    
    if stats['processed'] > 0:
        avg_confidence = stats['total_confidence'] / stats['processed']
        logging.info(f"  平均置信度: {avg_confidence:.3f}")


if __name__ == "__main__":
    main()
