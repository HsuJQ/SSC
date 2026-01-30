"""
Qwen Goal Interpretation - 常量定义
"""
API_URL = "http://127.0.0.1:1040/v1/chat/completions"
MODEL_NAME = "qwen2.5_7b_instruct"
PROMPT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome_m/generate_prompts"
OUTPUT_BASE_PATH = "/opt/pangu/examples/vllm-inference/virtualhome_m/llm_output_Qwen"
SYS_PROMPT = "You are Qwen, a helpful assistant. Please follow the user's instructions carefully and provide accurate responses in the requested format."
LOG_FILE = "client_generate_qwen_unknown.log"
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
