"""
Pangu Action Sequencing - PDDL/输出处理工具
"""
import re
import json
from typing import Optional, Dict, List, Tuple
from .constants import VALID_ACTIONS

def parse_output(output_sent: Optional[str]) -> str:
    if output_sent is None:
        return ""
    if "[unused17]" in output_sent:
        content = output_sent.split("[unused17]")[-1].split("[unused10]")[0].strip()
    else:
        content = output_sent.strip()
    return content

def extract_json_from_output(raw_output: str) -> Optional[Dict]:
    if not raw_output:
        return None
    output = raw_output.strip()
    output = re.sub(r'^```json\s*', '', output, flags=re.MULTILINE)
    output = re.sub(r'^```\s*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'```$', '', output)
    output = output.strip()
    json_match = re.search(r'(\{[\s\S]*\})', output)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return None

def normalize_action_sequence(data: Dict) -> Optional[List[Tuple[str, List]]]:
    if not data or not isinstance(data, dict):
        return None
    action_sequence = []
    for action_name, args in data.items():
        normalized_action = action_name.upper()
        if normalized_action not in VALID_ACTIONS:
            continue
        if args is None:
            args = []
        elif not isinstance(args, list):
            args = [args]
        action_sequence.append((normalized_action, args))
    return action_sequence if action_sequence else None
