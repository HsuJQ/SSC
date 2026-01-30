"""PDDL/subgoal utilities for Pangu SSC."""
import re
import json

def extract_predicates_from_subgoal(subgoal, valid_predicates):
    pattern = r'([A-Z][A-Z_]+)\s*\('
    matches = re.findall(pattern, subgoal)
    return [m for m in matches if m in valid_predicates]

def extract_json_from_output(raw_output):
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

def validate_subgoal_output(data):
    if not data or not isinstance(data, dict):
        return False
    if "output" not in data:
        return False
    if not isinstance(data.get("output"), list):
        return False
    return True

def normalize_subgoal_sequence(data):
    if not validate_subgoal_output(data):
        return None
    output_list = data.get("output", [])
    if not output_list:
        return None
    normalized_subgoals = []
    for subgoal in output_list:
        if isinstance(subgoal, str):
            cleaned = subgoal.strip()
            if cleaned:
                normalized_subgoals.append(cleaned)
    return normalized_subgoals if normalized_subgoals else None
