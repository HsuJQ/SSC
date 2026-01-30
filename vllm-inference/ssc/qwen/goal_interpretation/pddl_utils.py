"""
Qwen Goal Interpretation - 输出处理与验证
"""
import re
import json
import logging
from .constants import VALID_ACTIONS, VALID_RELATIONS, VALID_STATES

def parse_output(output_sent):
    if output_sent is None:
        return ""
    return output_sent.strip()

def enhanced_parse_output(raw_output):
    if raw_output is None:
        return ""
    output = raw_output.strip()
    output = re.sub(r'^```json\s*', '', output, flags=re.MULTILINE)
    output = re.sub(r'^```\s*$', '', output, flags=re.MULTILINE)
    output = re.sub(r'```$', '', output)
    output = output.strip()
    json_match = re.search(r'(\{[\s\S]*\})', output)
    if json_match:
        output = json_match.group(1)
    output = re.sub(r'"node_goals"', '"node goals"', output)
    output = re.sub(r'"edge_goals"', '"edge goals"', output)
    output = re.sub(r'"action_goals"', '"action goals"', output)
    output = re.sub(r'"node_gories"', '"node goals"', output)
    output = re.sub(r'"edge_gories"', '"edge goals"', output)
    output = re.sub(r'"action_gories"', '"action goals"', output)
    def fix_action_case(match):
        action_name = match.group(1)
        upper_action = action_name.upper()
        if upper_action in VALID_ACTIONS:
            return f'"action": "{upper_action}"'
        return match.group(0)
    output = re.sub(r'"action":\s*"([^"]+)"', fix_action_case, output, flags=re.IGNORECASE)
    return output.strip()

def validate_output(llm_output):
    if not llm_output:
        return False, "Empty output", None
    try:
        data = json.loads(llm_output)
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)[:100]}", None
    required_keys = ["node goals", "edge goals", "action goals"]
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return False, f"Missing keys: {missing_keys}", None
    node_goals = data.get("node goals", [])
    if not isinstance(node_goals, list):
        return False, "'node goals' should be a list", None
    for i, node in enumerate(node_goals):
        if not isinstance(node, dict):
            return False, f"node goals[{i}] is not a dict", None
        if 'name' not in node:
            return False, f"node goals[{i}] missing 'name'", None
        if 'state' not in node:
            return False, f"node goals[{i}] missing 'state'", None
    edge_goals = data.get("edge goals", [])
    if not isinstance(edge_goals, list):
        return False, "'edge goals' should be a list", None
    for i, edge in enumerate(edge_goals):
        if not isinstance(edge, dict):
            return False, f"edge goals[{i}] is not a dict", None
        if 'from_name' not in edge:
            return False, f"edge goals[{i}] missing 'from_name'", None
        if 'relation' not in edge:
            return False, f"edge goals[{i}] missing 'relation'", None
        if 'to_name' not in edge:
            return False, f"edge goals[{i}] missing 'to_name'", None
    action_goals = data.get("action goals", [])
    if not isinstance(action_goals, list):
        return False, "'action goals' should be a list", None
    for i, action in enumerate(action_goals):
        if not isinstance(action, dict):
            return False, f"action goals[{i}] is not a dict", None
        if 'action' not in action:
            return False, f"action goals[{i}] missing 'action'", None
        if 'description' not in action:
            return False, f"action goals[{i}] missing 'description'", None
    return True, "Valid", data
