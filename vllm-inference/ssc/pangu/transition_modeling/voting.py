"""Self-Consistency voting and signature utilities for Pangu SSC."""
import hashlib
from collections import Counter
from .pddl_utils import validate_transition_output, extract_pddl_structure

def get_action_skeleton_signature(data, valid_predicates):
    if not validate_transition_output(data):
        return ""
    pddl_output = data.get("output", "")
    structure = extract_pddl_structure(pddl_output, valid_predicates)
    if not structure.get("action_names"):
        return ""
    actions = "|".join(sorted(structure["action_names"]))
    predicates = ",".join(sorted(structure["predicates_used"]))
    signature = f"{actions}:{predicates}"
    features = []
    if structure["has_or"]:
        features.append("OR")
    if structure["has_when"]:
        features.append("WHEN")
    if structure["has_forall"]:
        features.append("FORALL")
    if structure["has_exists"]:
        features.append("EXISTS")
    if features:
        signature += f"[{','.join(features)}]"
    return signature

def get_full_signature(data):
    if not data:
        return ""
    pddl_output = data.get("output", "")
    if not pddl_output:
        return ""
    import re
    normalized = re.sub(r'\s+', ' ', pddl_output.lower()).strip()
    return hashlib.md5(normalized.encode()).hexdigest()

def get_weighted_signature(data, valid_predicates):
    if not validate_transition_output(data):
        return ""
    pddl_output = data.get("output", "")
    structure = extract_pddl_structure(pddl_output, valid_predicates)
    action_count = structure.get("action_count", 0)
    predicate_count = len(structure.get("predicates_used", []))
    complexity_score = 0
    if structure.get("has_or"):
        complexity_score += 1
    if structure.get("has_when"):
        complexity_score += 1
    if structure.get("has_forall"):
        complexity_score += 2
    if structure.get("has_exists"):
        complexity_score += 2
    if complexity_score <= 1:
        complexity = "simple"
    elif complexity_score <= 3:
        complexity = "moderate"
    else:
        complexity = "complex"
    skeleton = get_action_skeleton_signature(data, valid_predicates)
    return f"{complexity}-{action_count}-{predicate_count}-{skeleton[:50]}"

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
                sig = get_action_skeleton_signature(candidate, self.valid_predicates)
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
            "winning_signature": most_common_sig[:80] + "..." if len(most_common_sig) > 80 else most_common_sig
        }
        return best_result, confidence, vote_details
