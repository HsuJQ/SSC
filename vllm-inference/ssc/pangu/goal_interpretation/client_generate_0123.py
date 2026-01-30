import os
import json
import glob
import requests
import time
import logging
import psutil
import torch
import torch_npu
import re
from urllib.parse import urlparse
from tqdm import tqdm

# 配置部分
API_URL = "http://127.0.0.1:1040/v1/chat/completions"
MODEL_NAME = "pangu_embedded_7b"

def detect_server_info():
    model_path = "Unknown"
    device_ids = []
    try:
        parsed = urlparse(API_URL)
        if parsed.port:
            for conn in psutil.net_connections():
                if conn.laddr.port == parsed.port and conn.status == 'LISTEN':
                    proc = psutil.Process(conn.pid)
                    cmdline = proc.cmdline()
                    
                    # Detect Model Path
                    if 'serve' in cmdline:
                        try:
                            idx = cmdline.index('serve')
                            if idx + 1 < len(cmdline) and not cmdline[idx+1].startswith('-'):
                                model_path = cmdline[idx+1]
                        except ValueError:
                            pass
                    if model_path == "Unknown" and '--model' in cmdline:
                         try:
                            idx = cmdline.index('--model')
                            if idx + 1 < len(cmdline):
                                model_path = cmdline[idx+1]
                         except ValueError:
                            pass
                    
                    # Detect Devices
                    environ = proc.environ()
                    if 'ASCEND_RT_VISIBLE_DEVICES' in environ:
                        devs = environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
                        device_ids = [d.strip() for d in devs if d.strip()]
                    break
    except Exception as e:
        print(f"Warning: Failed to detect server info: {e}")
    
    return model_path, device_ids

# 自动检测
DETECTED_MODEL_PATH, DETECTED_DEVICE_IDS = detect_server_info()
DEVICE_NAME = "".join(DETECTED_DEVICE_IDS) if DETECTED_DEVICE_IDS else "unknown"
LOG_FILE = f"client_generate_{DEVICE_NAME}.log"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - %(levelname)s - [{DEVICE_NAME}] - [{MODEL_NAME}] - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def log_npu_info():
    try:
        logging.info(f"检测到 {torch.npu.device_count()} 个 NPU 设备 (Client可见)")
        
        if DETECTED_DEVICE_IDS:
            logging.info(f"Pangu 模型运行在 NPU: {', '.join(DETECTED_DEVICE_IDS)}")
            for dev_id in DETECTED_DEVICE_IDS:
                try:
                    idx = int(dev_id)
                    name = torch.npu.get_device_name(idx)
                    total_mem = torch.npu.get_device_properties(idx).total_memory / (1024**3)
                    logging.info(f"  设备 {dev_id}: {name}, 总显存: {total_mem:.2f} GB")
                except Exception as e:
                    logging.warning(f"  无法获取设备 {dev_id} 信息: {e}")
        else:
            logging.warning("未能检测到 Pangu 模型运行的 NPU 设备 (可能无法访问服务器进程信息)")

        logging.info(f"Pangu 模型路径: {DETECTED_MODEL_PATH}")
    except Exception as e:
        logging.error(f"记录 NPU 信息时出错: {e}")

# 路径配置 (与原 generate.py 保持一致)
PROMPT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome_m/generate_prompts"
OUTPUT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome_m/llm_output"

# 系统提示词 (与原 generate.py 保持一致)
SYS_PROMPT = "你必须严格遵守法律法规和社会道德规范。" \
    "生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。" \
    "一旦检测到输入或输出有此类倾向，应拒绝回答并发出警告。例如，如果输入内容包含暴力威胁或色情描述，" \
    "应返回错误信息：“您的输入包含不当内容，无法处理。”"

def call_vllm_api(prompt_text):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 8192,  # 调整为 8192 以避免超出 16384 的上下文限制 (16384 - ~5000 prompt)
        "stop_token_ids": [45892], # 对应 eos_token_id=45892
        "temperature": 0.3, # 默认值
        "top_p": 0.9
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        return content
    except Exception as e:
        logging.error(f"API调用失败: {e}")
        if 'response' in locals() and response.status_code != 200:
            logging.error(f"服务器返回: {response.text}")
        return None

# ==================== 输出验证配置 ====================
# 有效的状态值集合
VALID_STATES = {
    'CLOSED', 'OPEN', 'ON', 'OFF', 'SITTING', 'DIRTY', 'CLEAN', 
    'LYING', 'PLUGGED_IN', 'PLUGGED_OUT'
}

# 有效的关系值集合
VALID_RELATIONS = {'ON', 'INSIDE', 'BETWEEN', 'CLOSE', 'FACING', 'HOLDS_RH', 'HOLDS_LH'}

# 有效的动作集合
VALID_ACTIONS = {
    'CLOSE', 'DRINK', 'FIND', 'WALK', 'GRAB', 'LOOKAT', 'LOOKAT_SHORT', 
    'LOOKAT_LONG', 'OPEN', 'POINTAT', 'PUTBACK', 'PUTIN', 'PUTOBJBACK', 
    'RUN', 'SIT', 'STANDUP', 'SWITCHOFF', 'SWITCHON', 'TOUCH', 'TURNTO', 
    'WATCH', 'WIPE', 'PUTON', 'PUTOFF', 'GREET', 'DROP', 'READ', 'LIE', 
    'POUR', 'TYPE', 'PUSH', 'PULL', 'MOVE', 'WASH', 'RINSE', 'SCRUB', 
    'SQUEEZE', 'PLUGIN', 'PLUGOUT', 'CUT', 'EAT', 'RELEASE'
}

def parse_output(output_sent):
    """基础后处理：处理特殊token"""
    if output_sent is None:
        return ""
        
    if "[unused17]" in output_sent:
        content = output_sent.split("[unused17]")[-1].split("[unused10]")[0].strip()
    else:
        content = output_sent.strip()
    return content

def enhanced_parse_output(raw_output):
    """
    增强的输出后处理逻辑：
    1. 移除代码块标记 (```json)
    2. 提取JSON部分（跳过推理文本）
    3. 统一键名格式 (node_goals -> node goals)
    4. 修复action大小写问题
    5. 移除中文字符相关的错误键
    """
    if raw_output is None:
        return ""
    
    output = raw_output.strip()
    
    # 1. 移除代码块标记
    output = re.sub(r'^```json\s*', '', output, flags=re.MULTILINE)
    output = re.sub(r'^```\s*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'```$', '', output)
    output = output.strip()
    
    # 2. 提取JSON部分（跳过推理文本）
    # 查找第一个 { 和最后一个 } 之间的内容
    json_match = re.search(r'(\{[\s\S]*\})', output)
    if json_match:
        output = json_match.group(1)
    
    # 3. 统一键名格式：下划线格式 -> 空格格式
    output = re.sub(r'"node_goals"', '"node goals"', output)
    output = re.sub(r'"edge_goals"', '"edge goals"', output)
    output = re.sub(r'"action_goals"', '"action goals"', output)
    
    # 修复常见的拼写错误
    output = re.sub(r'"node_gories"', '"node goals"', output)
    output = re.sub(r'"edge_gories"', '"edge goals"', output)
    output = re.sub(r'"action_gories"', '"action goals"', output)
    
    # 4. 修复action大小写问题
    def fix_action_case(match):
        action_name = match.group(1)
        upper_action = action_name.upper()
        if upper_action in VALID_ACTIONS:
            return f'"action": "{upper_action}"'
        return match.group(0)
    
    output = re.sub(r'"action":\s*"([^"]+)"', fix_action_case, output, flags=re.IGNORECASE)
    
    # 5. 尝试移除包含中文的无效键（如 "edge立刻关系"）
    # 这个比较复杂，我们在验证层处理
    
    return output.strip()

def validate_output(llm_output):
    """
    验证输出格式和内容合法性
    
    Returns:
        (is_valid: bool, error_msg: str, parsed_data: dict or None)
    """
    if not llm_output:
        return False, "Empty output", None
    
    # 1. 尝试解析JSON
    try:
        data = json.loads(llm_output)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)[:100]}", None
    
    # 2. 检查必要键
    required_keys = ["node goals", "edge goals", "action goals"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return False, f"Missing keys: {missing_keys}", None
    
    # 3. 验证 node goals
    node_goals = data.get("node goals", [])
    if not isinstance(node_goals, list):
        return False, "'node goals' should be a list", None
    
    for i, node in enumerate(node_goals):
        if not isinstance(node, dict):
            return False, f"node goals[{i}] is not a dict", None
        if 'name' not in node:
            return False, f"node goals[{i}] missing 'name'", None
        if 'state' not in node:
            return False, f"node goals[{i}] missing 'state'", None
        # 状态值验证（宽松模式：只警告不拒绝）
        # if node.get('state') not in VALID_STATES:
        #     logging.warning(f"Invalid state: {node.get('state')}")
    
    # 4. 验证 edge goals
    edge_goals = data.get("edge goals", [])
    if not isinstance(edge_goals, list):
        return False, "'edge goals' should be a list", None
    
    for i, edge in enumerate(edge_goals):
        if not isinstance(edge, dict):
            return False, f"edge goals[{i}] is not a dict", None
        if 'from_name' not in edge:
            return False, f"edge goals[{i}] missing 'from_name'", None
        if 'relation' not in edge:
            return False, f"edge goals[{i}] missing 'relation'", None
        if 'to_name' not in edge:
            return False, f"edge goals[{i}] missing 'to_name'", None
        # 关系值验证（宽松模式）
        # if edge.get('relation') not in VALID_RELATIONS:
        #     logging.warning(f"Invalid relation: {edge.get('relation')}")
    
    # 5. 验证 action goals
    action_goals = data.get("action goals", [])
    if not isinstance(action_goals, list):
        return False, "'action goals' should be a list", None
    
    for i, action in enumerate(action_goals):
        if not isinstance(action, dict):
            return False, f"action goals[{i}] is not a dict", None
        if 'action' not in action:
            return False, f"action goals[{i}] missing 'action'", None
        if 'description' not in action:
            return False, f"action goals[{i}] missing 'description'", None
    
    return True, "Valid", data

def clean_and_fix_json(llm_output):
    """
    尝试清理和修复JSON输出中的问题
    """
    if not llm_output:
        return llm_output
    
    try:
        data = json.loads(llm_output)
        
        # 移除包含中文字符的无效键
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        keys_to_remove = [key for key in data.keys() if chinese_pattern.search(key)]
        for key in keys_to_remove:
            del data[key]
            logging.debug(f"Removed invalid key with Chinese chars: {key}")
        
        # 确保必要键存在
        if "node goals" not in data:
            data["node goals"] = []
        if "edge goals" not in data:
            data["edge goals"] = []
        if "action goals" not in data:
            data["action goals"] = []
        
        # 修复 edge goals 中的无效 relation
        valid_edges = []
        for edge in data.get("edge goals", []):
            if isinstance(edge, dict) and edge.get('relation') in VALID_RELATIONS:
                valid_edges.append(edge)
        data["edge goals"] = valid_edges
        
        # 修复 action goals 中的大小写
        for action in data.get("action goals", []):
            if isinstance(action, dict) and 'action' in action:
                upper_action = action['action'].upper()
                if upper_action in VALID_ACTIONS:
                    action['action'] = upper_action
        
        return json.dumps(data, ensure_ascii=False)
    except:
        return llm_output

def call_vllm_api_with_retry(prompt_text, max_retries=3, backoff_factor=2):
    """
    带验证的重试生成机制
    
    Args:
        prompt_text: 输入提示
        max_retries: 最大重试次数
        backoff_factor: 退避因子
    
    Returns:
        (final_output: str, is_valid: bool, attempts: int)
    """
    last_output = ""
    last_error = "No attempts made"
    
    for attempt in range(max_retries):
        # 调用API
        raw_output = call_vllm_api(prompt_text)
        
        if raw_output is None:
            last_error = "API call returned None"
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                logging.warning(f"Attempt {attempt+1}/{max_retries} failed (API error), retrying in {wait_time}s...")
                time.sleep(wait_time)
            continue
        
        # 基础后处理
        basic_output = parse_output(raw_output)
        
        # 增强后处理
        enhanced_output = enhanced_parse_output(basic_output)
        
        # 验证输出
        is_valid, error_msg, parsed_data = validate_output(enhanced_output)
        
        if is_valid:
            logging.debug(f"Valid output on attempt {attempt+1}")
            return enhanced_output, True, attempt + 1
        
        # 尝试修复
        fixed_output = clean_and_fix_json(enhanced_output)
        is_valid_after_fix, _, _ = validate_output(fixed_output)
        
        if is_valid_after_fix:
            logging.debug(f"Output fixed and valid on attempt {attempt+1}")
            return fixed_output, True, attempt + 1
        
        last_output = enhanced_output
        last_error = error_msg
        
        if attempt < max_retries - 1:
            wait_time = backoff_factor ** attempt
            logging.warning(f"Attempt {attempt+1}/{max_retries} failed ({error_msg}), retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    # 所有重试失败，返回最后一次的输出（可能不完美但有值）
    logging.warning(f"All {max_retries} attempts failed. Last error: {last_error}")
    return last_output if last_output else "", False, max_retries

def main():
    # 记录 NPU 信息
    log_npu_info()

    # 创建输出目录
    os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)

    # 查找任务目录
    if not os.path.exists(PROMPT_BASE_PATH):
        logging.error(f"错误: 找不到输入路径 {PROMPT_BASE_PATH}")
        return

    task_dirs = [d for d in os.listdir(PROMPT_BASE_PATH) if os.path.isdir(os.path.join(PROMPT_BASE_PATH, d))]
    
    if not task_dirs:
        logging.warning(f"在 {PROMPT_BASE_PATH} 中未找到任务目录")
        return

    for task in task_dirs:
        task_path = os.path.join(PROMPT_BASE_PATH, task)
        json_files = glob.glob(os.path.join(task_path, "*.json"))
        
        for json_file in json_files:
            logging.info(f"正在处理文件: {json_file}")
            
            # 构建输出文件名
            base_name = os.path.basename(json_file)
            output_name = base_name.replace("_prompts.json", "_outputs.json")
            if output_name == base_name:
                 output_name = base_name.replace(".json", "_outputs.json")
            output_file = os.path.join(OUTPUT_BASE_PATH, output_name)

            # 读取输入
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    prompts = json.load(f)
            except Exception as e:
                logging.error(f"读取 {json_file} 失败: {e}")
                continue
            
            # 检查是否已有部分结果 (断点续传)
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
            
            # 统计信息
            stats = {
                "total": 0,
                "valid": 0,
                "invalid": 0,
                "retried": 0,
                "failed": 0
            }
            
            # 使用 tqdm 显示进度条
            for item in tqdm(prompts, desc=f"Processing {base_name}", unit="item"):
                identifier = item.get("identifier")
                llm_prompt = item.get("llm_prompt")
                
                if not identifier or not llm_prompt:
                    continue

                if identifier in processed_ids:
                    continue
                
                stats["total"] += 1
                
                # 使用带重试的API调用
                final_output, is_valid, attempts = call_vllm_api_with_retry(
                    llm_prompt, 
                    max_retries=3,
                    backoff_factor=2
                )
                
                # 更新统计
                if is_valid:
                    stats["valid"] += 1
                else:
                    stats["invalid"] += 1
                
                if attempts > 1:
                    stats["retried"] += 1
                
                if final_output:
                    result_item = {
                        "identifier": identifier,
                        "llm_output": final_output
                    }
                    results.append(result_item)
                    
                    # 每生成一条就保存一次，防止意外中断丢失
                    try:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, indent=4, ensure_ascii=False)
                    except Exception as e:
                        logging.error(f"保存文件失败: {e}")
                else:
                    stats["failed"] += 1
                    logging.warning(f"  {identifier} 生成失败，跳过。")
            
            # 输出统计信息
            logging.info(f"文件 {base_name} 处理完成:")
            logging.info(f"  总数: {stats['total']}, 有效: {stats['valid']}, 无效: {stats['invalid']}")
            logging.info(f"  重试: {stats['retried']}, 完全失败: {stats['failed']}")
            if stats['total'] > 0:
                success_rate = (stats['valid'] / stats['total']) * 100
                logging.info(f"  成功率: {success_rate:.1f}%")

    logging.info("所有任务完成。")

if __name__ == "__main__":
    main()
