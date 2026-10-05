"""
ZSGI Core: Fork-Token Localization

Identifies the first token within a step where the model's prediction is
both high-entropy AND high CFG-difference -- the "fork point" where the
reasoning path likely diverged toward an error.

Rollback granularity: intra-step token level (not whole-step, not whole-chain).
"""

from typing import List, Optional, Dict, Tuple


def locate_fork_token(
    token_entropies: List[float],
    token_instabilities: List[float],
    step_token_indices: List[int],
    entropy_threshold: float,
    instability_threshold: float,
) -> Optional[int]:
    """Find the first fork token within a step.

    A fork token is the first token in the step that satisfies BOTH:
      - entropy > entropy_threshold  (model is uncertain)
      - instability > instability_threshold  (CFG significantly shifts prediction)

    Args:
        token_entropies: per-token entropy values (length = total generated tokens)
        token_instabilities: per-token instability (CFG-difference) scores
        step_token_indices: token indices belonging to the current step
        entropy_threshold: minimum entropy to qualify as a fork
        instability_threshold: minimum instability to qualify as a fork

    Returns:
        the index (in generation-relative coordinates) of the first fork token,
        or None if no fork is found.
    """
    for idx in step_token_indices:
        if idx >= len(token_entropies) or idx >= len(token_instabilities):
            break
        if (
            token_entropies[idx] > entropy_threshold
            and token_instabilities[idx] > instability_threshold
        ):
            return idx
    return None


def locate_fork_token_ranked(
    token_entropies: List[float],
    token_instabilities: List[float],
    step_token_indices: List[int],
    top_k: int = 1,
) -> List[Tuple[int, float]]:
    """Alternative: rank tokens by combined score and return top-k fork candidates.

    Combined score = entropy * instability (both high => strong fork signal).

    Returns:
        list of (token_idx, score) sorted by score descending, length up to top_k
    """
    candidates = []
    for idx in step_token_indices:
        if idx >= len(token_entropies) or idx >= len(token_instabilities):
            break
        score = token_entropies[idx] * token_instabilities[idx]
        candidates.append((idx, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


class ForkLocator:
    """Stateful fork locator that accumulates per-token signals."""

    def __init__(
        self,
        entropy_threshold: float = 0.5,
        instability_threshold: float = 0.3,
    ):
        self.entropy_threshold = entropy_threshold
        self.instability_threshold = instability_threshold
        self.token_entropies: List[float] = []
        self.token_instabilities: List[float] = []

    def record(self, entropy: float, instability: float):
        self.token_entropies.append(entropy)
        self.token_instabilities.append(instability)

    def find_fork(self, step_token_indices: List[int]) -> Optional[int]:
        return locate_fork_token(
            self.token_entropies,
            self.token_instabilities,
            step_token_indices,
            self.entropy_threshold,
            self.instability_threshold,
        )

    def find_fork_ranked(self, step_token_indices: List[int], top_k: int = 1):
        return locate_fork_token_ranked(
            self.token_entropies,
            self.token_instabilities,
            step_token_indices,
            top_k=top_k,
        )
