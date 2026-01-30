import os
import json
import glob
import requests
import time
import logging
import psutil
import torch
import torch_npu
from urllib.parse import urlparse
from tqdm import tqdm

# 配置部分
API_URL = "http://127.0.0.1:1041/v1/chat/completions"
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
PROMPT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome/a_test/generate_prompts"
OUTPUT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome/a_test/llm_output"

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
        "temperature": 1.0, # 默认值
        "top_p": 1.0
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

def parse_output(output_sent):
    # 复刻原 generate.py 的后处理逻辑
    if output_sent is None:
        return ""
        
    if "[unused17]" in output_sent:
        content = output_sent.split("[unused17]")[-1].split("[unused10]")[0].strip()
    else:
        content = output_sent.strip()
    return content

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
            
            # 使用 tqdm 显示进度条
            for item in tqdm(prompts, desc=f"Processing {base_name}", unit="item"):
                identifier = item.get("identifier")
                llm_prompt = item.get("llm_prompt")
                
                if not identifier or not llm_prompt:
                    continue

                if identifier in processed_ids:
                    continue
                
                raw_output = call_vllm_api(llm_prompt)
                final_output = parse_output(raw_output)
                
                if raw_output is not None:
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
                    logging.warning(f"  {identifier} 生成失败，跳过。")

    logging.info("所有任务完成。")

if __name__ == "__main__":
    main()
