"""Self-Consistency voting and signature utilities for Qwen subgoal decomposition."""
import hashlib
from collections import Counter
from .pddl_utils import validate_subgoal_output, normalize_subgoal_sequence, extract_predicates_from_subgoal

def get_subgoal_skeleton_signature(data, valid_predicates):
    subgoals = normalize_subgoal_sequence(data)
    if not subgoals:
        return ""
    all_predicates = []
    for subgoal in subgoals:
        preds = extract_predicates_from_subgoal(subgoal, valid_predicates)
        all_predicates.extend(preds)
    if not all_predicates:
        return ""
    return "->".join(all_predicates)

def get_full_signature(data):
    if not data:
        return ""
    import json
    try:
        normalized = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(normalized.encode()).hexdigest()
    except:
        return ""

def get_weighted_signature(data, valid_predicates):
    subgoals = normalize_subgoal_sequence(data)
    if not subgoals:
        return ""
    skeleton = get_subgoal_skeleton_signature(data, valid_predicates)
    length = len(subgoals)
    if 4 <= length <= 8:
        weight_prefix = "optimal"
    elif length < 4:
        weight_prefix = "short"
    else:
        weight_prefix = "long"
    return f"{weight_prefix}-{length}-{skeleton}"

class SelfConsistencyVoter:
    def __init__(self, strategy="skeleton", valid_predicates=None):
        self.strategy = strategy
        self.valid_predicates = valid_predicates
    def vote(self, candidates):
        if not candidates:
            return {{}}, 0.0, {"error": "No candidates"}
        if len(candidates) == 1:
            return candidates[0], 1.0, {"single_candidate": True}
        signatures = []
        for candidate in candidates:
            if self.strategy == "exact":
                sig = get_full_signature(candidate)
            elif self.strategy == "skeleton":
                sig = get_subgoal_skeleton_signature(candidate, self.valid_predicates)
            else:
                sig = get_weighted_signature(candidate, self.valid_predicates)
            signatures.append(sig)
        valid_pairs = [(sig, i) for i, sig in enumerate(signatures) if sig]
        if not valid_pairs:
            return candidates[0], 0.2, {"error": "All signatures empty"}
        vote_counter = Counter([sig for sig, _ in valid_pairs])
        most_common_sig, vote_count = vote_counter.most_common(1)[0]
        confidence = vote_count / len(candidates)
        best_result = None
        for sig, idx in valid_pairs:
            if sig == most_common_sig:
                best_result = candidates[idx]
                break
        vote_details = {
            "total_candidates": len(candidates),
            "valid_candidates": len(valid_pairs),
            "winning_votes": vote_count,
            "confidence": confidence,
            "vote_distribution": dict(vote_counter),
            "strategy": self.strategy,
            "winning_signature": most_common_sig[:50] + "..." if len(most_common_sig) > 50 else most_common_sig
        }
        return best_result, confidence, vote_details
