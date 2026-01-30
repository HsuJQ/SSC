"""
Pangu Action Sequencing with Self-Consistency (Modular Entrypoint)
"""
import argparse
import os
import json
import logging
from tqdm import tqdm
from ssc.pangu.action_sequencing.constants import (
    DEFAULT_NUM_SAMPLES, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS,
    PROMPT_FILE, OUTPUT_BASE_PATH, LOG_FILE, MODEL_NAME
)
from ssc.pangu.action_sequencing.runner import self_consistency_generate

def main():
    parser = argparse.ArgumentParser(
        description='VirtualHome Action Sequencing with Self-Consistency (Pangu, Modular)'
    )
    parser.add_argument('--num_samples', type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument('--top_p', type=float, default=DEFAULT_TOP_P)
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument('--voting_strategy', type=str, default='action_sequence',
                        choices=['exact', 'action_sequence', 'action_order'])
    parser.add_argument('--prompt_file', type=str, default=PROMPT_FILE)
    parser.add_argument('--output_dir', type=str, default=OUTPUT_BASE_PATH)
    parser.add_argument('--save_meta', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )
    logging.info("=" * 60)
    logging.info("VirtualHome Action Sequencing - Self-Consistency (Pangu, Modular)")
    logging.info("=" * 60)
    logging.info(f"配置: 模型: {MODEL_NAME} 样本数: {args.num_samples} Temperature: {args.temperature} Top-p: {args.top_p} 投票策略: {args.voting_strategy} 提示词文件: {args.prompt_file} 输出目录: {args.output_dir}")
    logging.info("=" * 60)

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
    output_name = base_name.replace("_prompts.json", f"_sc{args.num_samples}_outputs.json")
    if output_name == base_name:
        output_name = base_name.replace(".json", f"_sc{args.num_samples}_outputs.json")
    output_file = os.path.join(args.output_dir, output_name)
    meta_output_file = output_file.replace("_outputs.json", "_meta.json") if args.save_meta else None
    logging.info(f"输出文件: {output_file}")
    existing_results, existing_meta, processed_ids = [], [], set()
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
    stats = {"total": 0, "processed": 0, "high_confidence": 0, "medium_confidence": 0, "low_confidence": 0, "failed": 0, "total_confidence": 0.0}
    for item in tqdm(prompts, desc="Processing", unit="item"):
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
            result_item = {"identifier": identifier, "llm_output": output_str}
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
    logging.info("=" * 60)
    logging.info("处理完成!")
    logging.info("=" * 60)
    logging.info(f"统计信息: 总数: {stats['total']} 成功处理: {stats['processed']} 高置信度 (>0.6): {stats['high_confidence']} 中置信度 (0.4-0.6): {stats['medium_confidence']} 低置信度 (<0.4): {stats['low_confidence']} 失败: {stats['failed']}")
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
