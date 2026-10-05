"""
ZSGI Core: CFG-Difference Signals (Zero-Cost Step Verification)

Reuses the unconditional-branch logits already computed by MTI's CFG
intervention as a zero-cost verification signal. No extra token generation,
no extra forward pass beyond what MTI already does.

Signals computed per high-entropy token:
  - top1_flip: whether CFG changes the top-1 token
  - top1_prob_change: |p_cfg(top1) - p_orig(top1)|
  - topk_jaccard: Jaccard distance between top-k sets before/after CFG
  - kl_div: KL(p_cfg || p_orig)
  - confidence_gap_change: (p1-p2)_cfg - (p1-p2)_orig
  - entropy_change: H_cfg - H_orig
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional


@torch.no_grad()
def compute_cfg_signals(
    orig_logits: torch.Tensor,
    cfg_logits: torch.Tensor,
    top_k: int = 10,
) -> Dict[str, torch.Tensor]:
    """Compute CFG-difference signals for a batch of tokens.

    Args:
        orig_logits: conditional-branch logits, shape [B, V] (raw, pre-softmax)
        cfg_logits:  CFG-merged logits,       shape [B, V] (raw, pre-softmax)
        top_k:       k for top-k Jaccard / top-k overlap signals

    Returns:
        dict of signal tensors, each shape [B]
    """
    orig_probs = F.softmax(orig_logits, dim=-1)
    cfg_probs = F.softmax(cfg_logits, dim=-1)

    B, V = orig_probs.shape

    # 1. Top-1 flip
    orig_top1 = orig_probs.argmax(dim=-1)
    cfg_top1 = cfg_probs.argmax(dim=-1)
    top1_flip = (orig_top1 != cfg_top1).float()

    # 2. Top-1 probability change
    orig_top1_prob = orig_probs.gather(1, orig_top1.unsqueeze(-1)).squeeze(-1)
    cfg_top1_prob = cfg_probs.gather(1, cfg_top1.unsqueeze(-1)).squeeze(-1)
    top1_prob_change = (cfg_top1_prob - orig_top1_prob).abs()

    # 3. Top-k Jaccard distance
    k = min(top_k, V)
    orig_topk_idx = orig_probs.topk(k, dim=-1).indices
    cfg_topk_idx = cfg_probs.topk(k, dim=-1).indices
    # Build one-hot sets
    orig_set = torch.zeros(B, V, device=orig_probs.device, dtype=torch.bool)
    cfg_set = torch.zeros(B, V, device=orig_probs.device, dtype=torch.bool)
    orig_set.scatter_(1, orig_topk_idx, True)
    cfg_set.scatter_(1, cfg_topk_idx, True)
    intersection = (orig_set & cfg_set).sum(dim=-1).float()
    union = (orig_set | cfg_set).sum(dim=-1).float()
    topk_jaccard = 1.0 - (intersection / union.clamp(min=1.0))

    # 4. KL divergence: KL(cfg || orig)  -- how much CFG shifts distribution
    #    Use log-space for numerical stability
    log_orig = F.log_softmax(orig_logits, dim=-1)
    log_cfg = F.log_softmax(cfg_logits, dim=-1)
    kl_div = (cfg_probs * (log_cfg - log_orig)).sum(dim=-1)

    # 5. Confidence gap change: (p1 - p2) difference
    orig_top2 = orig_probs.topk(2, dim=-1).values
    cfg_top2 = cfg_probs.topk(2, dim=-1).values
    orig_gap = orig_top2[:, 0] - orig_top2[:, 1]
    cfg_gap = cfg_top2[:, 0] - cfg_top2[:, 1]
    confidence_gap_change = (cfg_gap - orig_gap).abs()

    # 6. Entropy change
    orig_entropy = -(orig_probs * log_orig).sum(dim=-1)
    cfg_entropy = -(cfg_probs * log_cfg).sum(dim=-1)
    entropy_change = (cfg_entropy - orig_entropy).abs()

    return {
        "top1_flip": top1_flip,
        "top1_prob_change": top1_prob_change,
        "topk_jaccard": topk_jaccard,
        "kl_div": kl_div,
        "confidence_gap_change": confidence_gap_change,
        "entropy_change": entropy_change,
    }


def aggregate_step_signals(
    token_signals: Dict[str, list],
    step_token_indices: list,
) -> Dict[str, float]:
    """Aggregate per-token CFG signals into a step-level instability score.

    Args:
        token_signals: dict of signal_name -> list of per-token values
        step_token_indices: list of token indices belonging to this step

    Returns:
        dict of signal_name -> aggregated scalar (mean over step tokens)
    """
    result = {}
    for name, values in token_signals.items():
        step_vals = [values[i] for i in step_token_indices]
        if len(step_vals) == 0:
            result[name] = 0.0
        else:
            result[name] = float(sum(step_vals) / len(step_vals))
    return result


def compute_step_instability(
    step_signals: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Combine multiple signals into a single step-instability scalar.

    Default weighting gives equal importance to normalized signals.
    A higher score means the step is more unstable / more likely to be wrong.
    """
    if weights is None:
        # Default: equal weight on all signals
        weights = {
            "top1_flip": 1.0,
            "top1_prob_change": 1.0,
            "topk_jaccard": 1.0,
            "kl_div": 1.0,
            "confidence_gap_change": 1.0,
            "entropy_change": 1.0,
        }
    score = 0.0
    total_w = 0.0
    for name, w in weights.items():
        if name in step_signals:
            score += w * step_signals[name]
            total_w += w
    return score / total_w if total_w > 0 else 0.0
