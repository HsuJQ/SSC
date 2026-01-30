"""
Pangu Action Sequencing - 常量定义
"""

VALID_ACTIONS = {
    'CLOSE', 'DRINK', 'FIND', 'WALK', 'GRAB', 'LOOKAT', 'LOOKAT_SHORT', 
    'LOOKAT_LONG', 'OPEN', 'POINTAT', 'PUTBACK', 'PUTIN', 'PUTOBJBACK', 
    'RUN', 'SIT', 'STANDUP', 'SWITCHOFF', 'SWITCHON', 'TOUCH', 'TURNTO', 
    'WATCH', 'WIPE', 'PUTON', 'PUTOFF', 'GREET', 'DROP', 'READ', 'LIE', 
    'POUR', 'TYPE', 'PUSH', 'PULL', 'MOVE', 'WASH', 'RINSE', 'SCRUB', 
    'SQUEEZE', 'PLUGIN', 'PLUGOUT', 'CUT', 'EAT', 'RELEASE'
}

API_URL = "http://127.0.0.1:1040/v1/chat/completions"
MODEL_NAME = "pangu_embedded_7b"
SYS_PROMPT = "你必须严格遵守法律法规和社会道德规范。生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。一旦检测到输入或输出有此类倾向，应拒绝回答并发出警告。"
PROMPT_FILE = "/opt/pangu/examples/vllm-inference/virtualhome/test/generate_prompts/action_sequencing/virtualhome_action_sequencing_prompts.json"
OUTPUT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome/action_m"
DEFAULT_NUM_SAMPLES = 5
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 4096
LOG_FILE = "action_sequencing_self_consistency.log"
