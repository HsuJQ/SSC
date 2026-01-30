"""PDDL utilities for Pangu SSC."""
import re
import json

def extract_action_names(pddl_output):
    pattern = r'\(:action\s+(\w+)'
    return re.findall(pattern, pddl_output, re.IGNORECASE)

def extract_predicates_from_pddl(pddl_output, valid_predicates):
    pattern = r'\((\w+)\s+[?\w]+'
    matches = re.findall(pattern, pddl_output, re.IGNORECASE)
    return [m for m in set(matches) if m.lower() in valid_predicates]

def extract_pddl_structure(pddl_output, valid_predicates):
    output_lower = pddl_output.lower()
    return {
        "action_names": extract_action_names(pddl_output),
        "predicates_used": extract_predicates_from_pddl(pddl_output, valid_predicates),
        "has_or": " or " in output_lower or "(or " in output_lower,
        "has_when": " when " in output_lower or "(when " in output_lower,
        "has_forall": " forall " in output_lower or "(forall " in output_lower,
        "has_exists": " exists " in output_lower or "(exists " in output_lower,
        "action_count": len(extract_action_names(pddl_output))
    }

def validate_transition_output(data):
    if not data or not isinstance(data, dict):
        return False
    if "output" not in data:
        return False
    output_str = data.get("output", "")
    if not isinstance(output_str, str) or not output_str.strip():
        return False
    if ':action' not in output_str.lower():
        return False
    return True

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
