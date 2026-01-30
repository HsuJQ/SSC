"""SSC main runner for Qwen subgoal decomposition: sampling, voting, result integration."""
from .api import call_qwen_api
from .pddl_utils import extract_json_from_output, validate_subgoal_output
from .voting import SelfConsistencyVoter

def parse_output(output_sent):
    if output_sent is None:
        return ""
    return output_sent.strip()

def generate_multiple_samples(prompt_text, num_samples, api_args, valid_predicates):
    candidates = []
    for _ in range(num_samples):
        raw_output = call_qwen_api(prompt_text, **api_args)
        if raw_output:
            processed = parse_output(raw_output)
            parsed = extract_json_from_output(processed)
            if parsed and validate_subgoal_output(parsed):
                candidates.append(parsed)
    return candidates

def self_consistency_generate(prompt_text, num_samples, api_args, voting_strategy, valid_predicates):
    candidates = generate_multiple_samples(prompt_text, num_samples, api_args, valid_predicates)
    if not candidates:
        return "", {"error": "No valid candidates", "num_samples": num_samples}
    voter = SelfConsistencyVoter(strategy=voting_strategy, valid_predicates=valid_predicates)
    best_result, confidence, vote_details = voter.vote(candidates)
    meta_info = {
        "num_samples": num_samples,
        "valid_candidates": len(candidates),
        "confidence": confidence,
        "vote_details": vote_details
    }
    import json
    try:
        output_str = json.dumps(best_result, ensure_ascii=False)
    except:
        output_str = ""
    return output_str, meta_info
