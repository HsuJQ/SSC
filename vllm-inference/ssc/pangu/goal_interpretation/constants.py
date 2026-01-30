"""
Pangu Goal Interpretation - 常量定义
"""
API_URL = "http://127.0.0.1:1040/v1/chat/completions"
MODEL_NAME = "pangu_embedded_7b"
PROMPT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome_m/generate_prompts"
OUTPUT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome_m/llm_output"
SYS_PROMPT = "你必须严格遵守法律法规和社会道德规范。生成任何内容时，都应避免涉及暴力、色情、恐怖主义、种族歧视、性别歧视等不当内容。一旦检测到输入或输出有此类倾向，应拒绝回答并发出警告。例如，如果输入内容包含暴力威胁或色情描述，应返回错误信息：“您的输入包含不当内容，无法处理。”"
LOG_FILE = "client_generate_unknown.log"
VALID_STATES = {
    'CLOSED', 'OPEN', 'ON', 'OFF', 'SITTING', 'DIRTY', 'CLEAN', 
    'LYING', 'PLUGGED_IN', 'PLUGGED_OUT'
}
VALID_RELATIONS = {'ON', 'INSIDE', 'BETWEEN', 'CLOSE', 'FACING', 'HOLDS_RH', 'HOLDS_LH'}
VALID_ACTIONS = {
    'CLOSE', 'DRINK', 'FIND', 'WALK', 'GRAB', 'LOOKAT', 'LOOKAT_SHORT', 
    'LOOKAT_LONG', 'OPEN', 'POINTAT', 'PUTBACK', 'PUTIN', 'PUTOBJBACK', 
    'RUN', 'SIT', 'STANDUP', 'SWITCHOFF', 'SWITCHON', 'TOUCH', 'TURNTO', 
    'WATCH', 'WIPE', 'PUTON', 'PUTOFF', 'GREET', 'DROP', 'READ', 'LIE', 
    'POUR', 'TYPE', 'PUSH', 'PULL', 'MOVE', 'WASH', 'RINSE', 'SCRUB', 
    'SQUEEZE', 'PLUGIN', 'PLUGOUT', 'CUT', 'EAT', 'RELEASE'
}
