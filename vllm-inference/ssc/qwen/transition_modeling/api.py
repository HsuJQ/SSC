"""
API module for model inference (Qwen, Pangu, etc.)
"""
import requests
import logging

def call_qwen_api(prompt_text, model_name, sys_prompt, api_url, stop_token_ids, temperature, top_p, max_tokens):
    """
    Call Qwen API for inference.
    Args:
        prompt_text: User prompt
        model_name: Model name string
        sys_prompt: System prompt string
        api_url: API endpoint
        stop_token_ids: List of stop token ids
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Max tokens to generate
    Returns:
        content string or None
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_tokens,
        "stop_token_ids": stop_token_ids,
        "temperature": temperature,
        "top_p": top_p
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=300)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        return content
    except Exception as e:
        logging.error(f"API call failed: {e}")
        return None
