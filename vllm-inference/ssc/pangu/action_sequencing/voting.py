"""
Pangu Action Sequencing - Self-Consistency 投票
"""
from collections import Counter
import hashlib
import json
from typing import List, Dict, Tuple
from .pddl_utils import normalize_action_sequence

def get_action_sequence_signature(data: Dict) -> str:
    sequence = normalize_action_sequence(data)
    if not sequence:
        return ""
    action_names = [action[0] for action in sequence]
    return "->".join(action_names)

def get_full_signature(data: Dict) -> str:
    if not data:
        return ""
    try:
        normalized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(normalized.encode()).hexdigest()
    except:
        return ""

class SelfConsistencyVoter:
    def __init__(self, strategy: str = "action_sequence"):
        self.strategy = strategy
    def vote(self, candidates: List[Dict]) -> Tuple[Dict, float, Dict]:
        if not candidates:
            return {{}}, 0.0, {"error": "No candidates"}
        if len(candidates) == 1:
            return candidates[0], 1.0, {"single_candidate": True}
        signatures = []
        for candidate in candidates:
            if self.strategy == "exact":
                sig = get_full_signature(candidate)
            else:
                sig = get_action_sequence_signature(candidate)
            signatures.append(sig)
        vote_counter = Counter(signatures)
        most_common_sig, vote_count = vote_counter.most_common(1)[0]
        confidence = vote_count / len(candidates)
        best_result = None
        for i, sig in enumerate(signatures):
            if sig == most_common_sig:
                best_result = candidates[i]
                break
        vote_details = {
            "total_candidates": len(candidates),
            "winning_votes": vote_count,
            "confidence": confidence,
            "vote_distribution": dict(vote_counter),
            "strategy": self.strategy
        }
        return best_result, confidence, vote_details
