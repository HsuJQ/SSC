"""
Qwen Action Sequencing - 常量定义
"""

# 有效动作集（需根据实际任务补充）
VALID_ACTIONS = [
    "FIND", "PUTBACK", "PICKUP", "OPEN", "CLOSE", "SWITCHON", "SWITCHOFF",
    "GOTO", "WALK", "RUN", "GRASP", "RELEASE", "USE", "LOOKAT", "TOUCH"
]

# 默认参数
DEFAULT_NUM_SAMPLES = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 4096

# API配置
API_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen1.5-7B-Chat"
SYS_PROMPT = "You are a helpful assistant for VirtualHome action sequencing."

# 文件路径
PROMPT_FILE = "prompts.json"
OUTPUT_BASE_PATH = "outputs/"
LOG_FILE = "action_sequencing_sc.log"
