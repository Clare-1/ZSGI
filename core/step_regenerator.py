"""
ZSGI Core: Step Regeneration

Handles the mechanics of rolling back the KV cache to a fork token and
regenerating the remainder of the step.

Key operations:
  - Truncate input_ids to the fork token position
  - Truncate KV cache to match
  - Re-run generation from the fork with adjusted sampling params
  - Evaluate each candidate with the zero-cost CFG-difference signal
  - Select the most stable candidate
"""

import copy
import torch
from typing import Optional, List, Dict, Tuple

from .cfg_signal import compute_cfg_signals, compute_step_instability


def truncate_kv_cache(past_key_values, keep_length: int):
    """Truncate KV cache to keep only the first `keep_length` tokens.

    Works with HuggingFace DynamicCache and similar tuple-based caches.

    Args:
        past_key_values: the KV cache object
        keep_length: number of tokens to keep (prefix length up to & including fork)

    Returns:
        truncated KV cache (new object)
    """
    if past_key_values is None:
        return None

    if hasattr(past_key_values, "crop"):
        # DynamicCache in newer transformers
        return past_key_values.crop(keep_length)

    # Tuple-based cache: each element is (key, value) of shape [B, H, S, D]
    truncated = []
    for layer_idx in range(len(past_key_values)):
        k, v = past_key_values[layer_idx]
        k_trunc = k[:, :, :keep_length, :]
        v_trunc = v[:, :, :keep_length, :]
        truncated.append((k_trunc, v_trunc))
    return tuple(truncated)


def regenerate_step(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    past_key_values,
    fork_token_idx: int,
    step_end_token_idx: int,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    top_p: float = 0.9,
    guidance_scale: float = 100.0,
    entropy_threshold: float = -1.0,
    lightweight_negative_prompt: str = "OUTPUT ERROR",
    num_candidates: int = 1,
    cfg_signal_callback=None,
) -> Tuple[torch.Tensor, Dict]:
    """Regenerate a step from the fork token.

    Args:
        model: the HF causal LM
        tokenizer: the tokenizer
        input_ids: full sequence tensor [1, seq_len]
        past_key_values: KV cache for the full sequence
        fork_token_idx: index (in the full sequence) to roll back to
        step_end_token_idx: original step end index (for length reference)
        max_new_tokens: maximum tokens to generate for the step
        temperature: sampling temperature for regeneration
        top_p: top-p sampling
        guidance_scale: CFG guidance scale (reuse MTI's)
        entropy_threshold: CFG entropy threshold
        lightweight_negative_prompt: MTI negative prompt
        num_candidates: number of regeneration candidates to try
        cfg_signal_callback: optional callback(cfg_signals) for logging

    Returns:
        (best_input_ids, info_dict) where best_input_ids has the regenerated
        step appended after the fork token.
    """
    device = input_ids.device

    # Roll back to fork token
    keep_ids = input_ids[:, : fork_token_idx + 1]
    truncated_kv = truncate_kv_cache(past_key_values, fork_token_idx + 1)

    best_ids = None
    best_instability = float("inf")
    best_info = {}

    for cand in range(num_candidates):
        # Generate from the fork
        with torch.no_grad():
            gen_out = model.generate(
                input_ids=keep_ids,
                past_key_values=truncated_kv,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                guidance_scale=guidance_scale,
                entropy_threshold=entropy_threshold,
                lightweight_negative_prompt=lightweight_negative_prompt,
                tokenizer__=tokenizer,
                return_dict_in_generate=True,
                output_scores=True,
            )

        cand_ids = gen_out.sequences
        # Evaluate stability of the regenerated portion
        new_tokens = cand_ids[0, fork_token_idx + 1 :].tolist()

        instability = 0.0
        if cfg_signal_callback is not None:
            instability = cfg_signal_callback(cand_ids)
        else:
            # Fallback: use entropy of generated tokens as proxy
            if hasattr(gen_out, "scores") and gen_out.scores:
                entropies = []
                for score in gen_out.scores:
                    probs = torch.softmax(score, dim=-1)
                    ent = -(probs * torch.log(probs.clamp(min=1e-12))).sum(-1)
                    entropies.append(ent.item())
                instability = sum(entropies) / max(len(entropies), 1)

        if instability < best_instability:
            best_instability = instability
            best_ids = cand_ids
            best_info = {
                "candidate": cand,
                "instability": instability,
                "num_new_tokens": len(new_tokens),
            }

    return best_ids, best_info


def select_best_candidate(
    candidates: List[torch.Tensor],
    signal_scores: List[float],
) -> Tuple[torch.Tensor, int]:
    """Select the candidate with the lowest instability (most stable)."""
    best_idx = min(range(len(signal_scores)), key=lambda i: signal_scores[i])
    return candidates[best_idx], best_idx
