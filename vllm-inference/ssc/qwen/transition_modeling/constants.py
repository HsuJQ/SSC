"""
Constants for predicates, keywords, and system prompt.
"""
VALID_PREDICATES = {
    'closed', 'open', 'on', 'off', 'plugged_in', 'plugged_out',
    'sitting', 'lying', 'clean', 'dirty',
    'obj_ontop', 'ontop', 'on_char', 'inside_room', 'obj_inside', 'inside',
    'obj_next_to', 'next_to', 'between', 'facing', 'holds_rh', 'holds_lh',
    'grabbable', 'cuttable', 'can_open', 'readable', 'has_paper',
    'movable', 'pourable', 'cream', 'has_switch', 'lookable', 'has_plug',
    'drinkable', 'body_part', 'recipient', 'containers', 'cover_object',
    'surfaces', 'sittable', 'lieable', 'person', 'hangable', 'clothes', 'eatable'
}

PDDL_KEYWORDS = {
    ':action', ':parameters', ':precondition', ':effect',
    'and', 'or', 'not', 'when', 'forall', 'exists'
}

SYS_PROMPT = """You are an expert PDDL planner specializing in VirtualHome domain actions.\n... (省略, 可从原脚本粘贴完整内容)\n"""
