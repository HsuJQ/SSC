"""
Pangu Action Sequencing - vLLM API 调用
"""
import requests
import logging
from typing import Optional
from .constants import API_URL, MODEL_NAME, SYS_PROMPT

def call_vllm_api(prompt_text: str, temperature: float = 0.7, 
                   top_p: float = 0.95, max_tokens: int = 4096) -> Optional[str]:
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_tokens,
        "stop_token_ids": [45892],
        "temperature": temperature,
        "top_p": top_p
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        return content
    except requests.exceptions.Timeout:
        logging.error("API调用超时")
        return None
    except Exception as e:
        logging.error(f"API调用失败: {e}")
        return None
