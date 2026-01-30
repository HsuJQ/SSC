"""
Pangu Action Sequencing - SSC主流程
"""
import logging
import json
from typing import List, Dict, Tuple
from .constants import DEFAULT_NUM_SAMPLES, DEFAULT_TEMPERATURE, DEFAULT_TOP_P, DEFAULT_MAX_TOKENS
from .api import call_vllm_api
from .pddl_utils import parse_output, extract_json_from_output
from .voting import SelfConsistencyVoter

def generate_multiple_samples(prompt_text: str, num_samples: int = 5,
                               temperature: float = 0.7, top_p: float = 0.95,
                               max_tokens: int = 4096) -> List[Dict]:
    candidates = []
    for _ in range(num_samples):
        raw_output = call_vllm_api(prompt_text, temperature, top_p, max_tokens)
        if raw_output:
            processed = parse_output(raw_output)
            parsed = extract_json_from_output(processed)
            if parsed:
                candidates.append(parsed)
    return candidates

def self_consistency_generate(prompt_text: str, num_samples: int = 5,
                               temperature: float = 0.7, top_p: float = 0.95,
                               max_tokens: int = 4096,
                               voting_strategy: str = "action_sequence") -> Tuple[str, Dict]:
    candidates = generate_multiple_samples(
        prompt_text, num_samples, temperature, top_p, max_tokens
    )
    if not candidates:
        logging.warning("No valid candidates generated")
        return "", {"error": "No valid candidates", "num_samples": num_samples}
    voter = SelfConsistencyVoter(strategy=voting_strategy)
    best_result, confidence, vote_details = voter.vote(candidates)
    meta_info = {
        "num_samples": num_samples,
        "valid_candidates": len(candidates),
        "confidence": confidence,
        "vote_details": vote_details
    }
    try:
        output_str = json.dumps(best_result, ensure_ascii=False)
    except:
        output_str = ""
    return output_str, meta_info
