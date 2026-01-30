"""
Qwen Goal Interpretation - vLLM API 调用
"""
import requests
import logging
from .constants import API_URL, MODEL_NAME, SYS_PROMPT

def call_vllm_api(prompt_text):
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": 8192,
        "temperature": 0.3,
        "top_p": 0.9,
        "repetition_penalty": 1.05
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
