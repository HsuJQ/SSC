"""
Qwen Goal Interpretation - 主流程
"""
import os
import json
import logging
from tqdm import tqdm
from .constants import PROMPT_BASE_PATH, OUTPUT_BASE_PATH, LOG_FILE
from .api import call_vllm_api
from .pddl_utils import parse_output, enhanced_parse_output, validate_output

def call_vllm_api_with_retry(prompt_text, max_retries=3, backoff_factor=2):
    last_output = ""
    last_error = "No attempts made"
    for attempt in range(max_retries):
        raw_output = call_vllm_api(prompt_text)
        if raw_output is None:
            last_error = "API call returned None"
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                logging.warning(f"Attempt {attempt+1}/{max_retries} failed (API error), retrying in {wait_time}s...")
                time.sleep(wait_time)
            continue
        basic_output = parse_output(raw_output)
        enhanced_output = enhanced_parse_output(basic_output)
        is_valid, error_msg, parsed_data = validate_output(enhanced_output)
        if is_valid:
            logging.debug(f"Valid output on attempt {attempt+1}")
            return enhanced_output, True, attempt + 1
        last_output = enhanced_output
        last_error = error_msg
        if attempt < max_retries - 1:
            wait_time = backoff_factor ** attempt
            logging.warning(f"Attempt {attempt+1}/{max_retries} failed ({error_msg}), retrying in {wait_time}s...")
            time.sleep(wait_time)
    logging.warning(f"All {max_retries} attempts failed. Last error: {last_error}")
    return last_output if last_output else "", False, max_retries

def process_all_tasks():
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
    if not os.path.exists(PROMPT_BASE_PATH):
        logging.error(f"错误: 找不到输入路径 {PROMPT_BASE_PATH}")
        return
    task_dirs = [d for d in os.listdir(PROMPT_BASE_PATH) if os.path.isdir(os.path.join(PROMPT_BASE_PATH, d))]
    if not task_dirs:
        logging.warning(f"在 {PROMPT_BASE_PATH} 中未找到任务目录")
        return
    for task in task_dirs:
        task_path = os.path.join(PROMPT_BASE_PATH, task)
        json_files = [f for f in os.listdir(task_path) if f.endswith('.json')]
        for json_file in json_files:
            json_path = os.path.join(task_path, json_file)
            logging.info(f"正在处理文件: {json_path}")
            base_name = os.path.basename(json_file)
            output_name = base_name.replace("_prompts.json", "_outputs.json")
            if output_name == base_name:
                output_name = base_name.replace(".json", "_outputs.json")
            output_file = os.path.join(OUTPUT_BASE_PATH, output_name)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
            except Exception as e:
                logging.error(f"读取 {json_path} 失败: {e}")
                continue
            existing_results = []
            processed_ids = set()
            if os.path.exists(output_file):
                try:
                    with open(output_file, 'r', encoding='utf-8') as f:
                        existing_results = json.load(f)
                        for item in existing_results:
                            processed_ids.add(item.get("identifier"))
                    logging.info(f"  发现已有结果文件，包含 {len(existing_results)} 条记录，将跳过已完成项。")
                except json.JSONDecodeError:
                    logging.warning("  已有结果文件损坏，将重新开始。")
            results = existing_results
            stats = {"total": 0, "valid": 0, "invalid": 0, "retried": 0, "failed": 0}
            for item in tqdm(prompts, desc=f"Processing {base_name}", unit="item"):
                identifier = item.get("identifier")
                llm_prompt = item.get("llm_prompt")
                if not identifier or not llm_prompt:
                    continue
                if identifier in processed_ids:
                    continue
                stats["total"] += 1
                final_output, is_valid, attempts = call_vllm_api_with_retry(
                    llm_prompt, max_retries=3, backoff_factor=2
                )
                if is_valid:
                    stats["valid"] += 1
                else:
                    stats["invalid"] += 1
                if attempts > 1:
                    stats["retried"] += 1
                if final_output:
                    result_item = {"identifier": identifier, "llm_output": final_output}
                    results.append(result_item)
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=4, ensure_ascii=False)
                    except Exception as e:
                        logging.error(f"保存文件失败: {e}")
                else:
                    stats["failed"] += 1
                    logging.warning(f"  {identifier} 生成失败，跳过。")
            logging.info(f"文件 {base_name} 处理完成:")
            logging.info(f"  总数: {stats['total']}, 有效: {stats['valid']}, 无效: {stats['invalid']}")
            logging.info(f"  重试: {stats['retried']}, 完全失败: {stats['failed']}")
            if stats['total'] > 0:
                success_rate = (stats['valid'] / stats['total']) * 100
                logging.info(f"  成功率: {success_rate:.1f}%")
    logging.info("所有任务完成。")
