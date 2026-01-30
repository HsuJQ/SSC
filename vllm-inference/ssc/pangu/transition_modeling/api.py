"""API module for Pangu Embedded model inference."""
import requests
import logging

def call_pangu_api(prompt_text, model_name, sys_prompt, api_url, stop_token_ids, temperature, top_p, max_tokens):
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
