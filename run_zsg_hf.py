"""
ZSGI: Zero-Cost Step-Guided Intervention (HuggingFace prototype)

Based on MTI's run_hf.py. This version:
  1. Runs MTI's CFG intervention (token-level)
  2. Records CFG-difference signals at every high-entropy token (zero-cost)
  3. Detects step boundaries in the generated text
  4. [Validation mode] Dumps per-token signals + step correctness labels
     for the correlation analysis (Week 1 deliverable)

Usage:
  python run_zsg_hf.py --model <model_path> --dataset gsm8k --num_samples 100
  python run_zsg_hf.py --model <model_path> --interactive   # single prompt
"""

import argparse
import json
import os
import sys
import copy
import math
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from transformers.generation.utils import GenerationMixin
from transformers.generation.logits_process import (
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
    LogitsProcessorList,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.cfg_signal import (
    compute_cfg_signals,
    aggregate_step_signals,
    compute_step_instability,
)
from core.step_detector import StepDetector, tokenize_step_boundaries


def get_entropy(logits):
    """MTI's entropy function (kept identical for compatibility)."""
    B, S, E = logits.shape
    with torch.no_grad():
        probs = F.softmax(logits, dim=-1)
        entropy_list = None
        for i in range(B):
            probs_chunk = probs[i : i + 1]
            entropy__ = probs_chunk * torch.log(probs_chunk)
            entropy_ = torch.where(
                ~torch.isnan(entropy__),
                entropy__,
                torch.zeros_like(entropy__),
            )
            entropy = -torch.sum(entropy_, dim=-1)
            entropy_list = (
                entropy if entropy_list is None else torch.cat([entropy_list, entropy], dim=0)
            )
    return entropy_list


# Global signal buffer: processor writes here, main loop reads after generate()
GLOBAL_SIGNAL_BUFFER: List[Dict[str, float]] = []


class ZSGIProcessor(UnbatchedClassifierFreeGuidanceLogitsProcessor):
    """MTI processor extended with zero-cost CFG-difference signal recording."""

    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: Optional[torch.LongTensor] = None,
        unconditional_attention_mask: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        entropy_threshold=None,
        tokenizer__=None,
        lightweight_negative_prompt=None,
    ):
        super().__init__(
            guidance_scale,
            model,
            unconditional_ids=unconditional_ids,
            unconditional_attention_mask=unconditional_attention_mask,
            use_cache=use_cache,
        )
        self.entropy_threshold = entropy_threshold
        self.tokenizer__ = tokenizer__
        self.lightweight_negative_prompt = lightweight_negative_prompt
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids):
        """MTI's unconditional branch computation (unchanged)."""
        mask_ = self.model.cfg_mask__[:, -1]
        uc_ = self.tokenizer__.batch_encode_plus(
            [self.lightweight_negative_prompt] * input_ids.shape[0]
        )
        uc_ = {k: torch.tensor(v).to(self.model.device) for k, v in uc_.items()}
        uc_input_ids_, uc_attention_mask_ = uc_["input_ids"], uc_["attention_mask"]

        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = torch.ones_like(
                    self.unconditional_context["input_ids"], dtype=torch.long
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = torch.cat(
                [
                    self.unconditional_context["attention_mask"],
                    torch.ones_like(input_ids[:, -1:], dtype=torch.long),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = torch.cat(
                    [self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1
                )
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        if mask_.sum() == 0:
            return None

        indices_ = mask_.nonzero(as_tuple=False).squeeze(1)
        kv_cache_ = copy.deepcopy(self.kv_cache_)
        kv_cache_.batch_select_indices(indices_)

        attention_mask_c_uc_ = torch.cat(
            [self.unconditional_context["attention_mask"], uc_attention_mask_], dim=1
        )[mask_]
        position_ids_ = (
            attention_mask_c_uc_.long().cumsum(-1) - 1
        ).masked_fill_(attention_mask_c_uc_ == 0, 1)
        position_ids_ = position_ids_[:, -uc_input_ids_[mask_].shape[-1] :]
        out = self.model(
            uc_input_ids_[mask_],
            attention_mask=attention_mask_c_uc_,
            use_cache=False,
            past_key_values=kv_cache_,
            position_ids=position_ids_,
        )
        del kv_cache_
        return out.logits

    def __call__(self, input_ids, scores, kv_cache_=None):
        self.kv_cache_ = kv_cache_
        if self.guidance_scale == 1:
            scores = torch.nn.functional.log_softmax(scores, dim=-1)
            self._record_signals_none()
            return scores

        c_entropy = get_entropy(scores.unsqueeze(1))
        mask_ = c_entropy > self.entropy_threshold
        self.model.cfg_mask__ = (
            mask_
            if self.model.cfg_mask__ is None
            else torch.cat([self.model.cfg_mask__, mask_], dim=-1)
        )

        orig_logits = scores.clone()  # conditional branch (before CFG), raw logits
        scores = torch.nn.functional.log_softmax(scores, dim=-1)
        uncond_raw_logits = self.get_unconditional_logits(input_ids)

        if uncond_raw_logits is None:
            self._record_signals_none()
            return scores

        unconditional_logits = torch.nn.functional.log_softmax(uncond_raw_logits[:, -1], dim=-1)
        unconditional_logits_ = torch.zeros_like(scores, dtype=unconditional_logits.dtype)
        mask_1d = mask_.squeeze(-1)  # [B] instead of scalar when B==1
        unconditional_logits_[mask_1d] = unconditional_logits
        guidance_scale = torch.where(
            mask_,
            self.guidance_scale * torch.ones_like(mask_),
            torch.ones_like(mask_),
        )
        scores_processed = (
            guidance_scale * (scores - unconditional_logits_) + unconditional_logits_
        )
        del unconditional_logits_

        # ZSGI: compute zero-cost CFG-difference signals in raw-logit space.
        # Reconstruct CFG logits from raw cond/uncond logits for a fair comparison.
        uncond_raw_last = uncond_raw_logits[:, -1, :].to(orig_logits.dtype)  # [B, V]
        uncond_for_cfg = torch.zeros_like(orig_logits)
        uncond_for_cfg[mask_1d] = uncond_raw_last
        cfg_scale_raw = torch.where(
            mask_1d.unsqueeze(-1),
            self.guidance_scale,
            1.0,
        )
        cfg_logits_raw = (
            cfg_scale_raw * (orig_logits - uncond_for_cfg) + uncond_for_cfg
        )
        self._record_cfg_signals(
            orig_logits=orig_logits[mask_1d],
            cfg_logits=cfg_logits_raw[mask_1d],
            entropy=c_entropy[mask_1d],
        )

        return scores_processed

    def _record_signals_none(self):
        """Record zero signal for tokens that didn't trigger CFG."""
        GLOBAL_SIGNAL_BUFFER.append(
            {
                "top1_flip": 0.0,
                "top1_prob_change": 0.0,
                "topk_jaccard": 0.0,
                "kl_div": 0.0,
                "confidence_gap_change": 0.0,
                "entropy_change": 0.0,
                "entropy": 0.0,
                "cfg_triggered": False,
            }
        )

    @torch.no_grad()
    def _record_cfg_signals(self, orig_logits, cfg_logits, entropy):
        """Record CFG-difference signals for the current batch of tokens."""
        if orig_logits.dim() == 1:
            orig_logits = orig_logits.unsqueeze(0)
            cfg_logits = cfg_logits.unsqueeze(0)
            entropy = entropy.unsqueeze(0)

        signals = compute_cfg_signals(orig_logits, cfg_logits)
        for i in range(orig_logits.shape[0]):
            GLOBAL_SIGNAL_BUFFER.append(
                {
                    "top1_flip": signals["top1_flip"][i].item(),
                    "top1_prob_change": signals["top1_prob_change"][i].item(),
                    "topk_jaccard": signals["topk_jaccard"][i].item(),
                    "kl_div": signals["kl_div"][i].item(),
                    "confidence_gap_change": signals["confidence_gap_change"][i].item(),
                    "entropy_change": signals["entropy_change"][i].item(),
                    "entropy": entropy[i].item(),
                    "cfg_triggered": True,
                }
            )


# ---- Monkey-patched generation (copied from MTI, with ZSGI signal hooks) ----

def _sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
):
    """MTI's _sample with ZSGI signal recording support."""
    pad_token_id = generation_config._pad_token_tensor
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(
        hasattr(criteria, "eos_token_id") for criteria in stopping_criteria
    )
    do_sample = generation_config.do_sample

    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None

    batch_size, cur_len = input_ids.shape[:2]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

    model_forward = self.__call__

    self.cfg_mask__ = None

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model_forward(**model_inputs, return_dict=True)
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        next_token_logits = outputs.logits[:, -1, :].to(
            copy=True, dtype=torch.float32, device=input_ids.device
        )
        self.position_ids_ = model_inputs["position_ids"]
        kwargs = {"kv_cache_": outputs.past_key_values}
        next_token_scores = logits_processor(input_ids, next_token_logits, **kwargs)

        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)

        if do_sample:
            probs = F.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                1 - unfinished_sequences
            )

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        return {
            "sequences": input_ids,
            "scores": scores,
            "logits": raw_logits,
        }
    return input_ids


def _get_logits_processor(
    self,
    generation_config: GenerationConfig,
    input_ids_seq_length=None,
    encoder_input_ids=None,
    prefix_allowed_tokens_fn=None,
    logits_processor=None,
    device=None,
    model_kwargs=None,
    negative_prompt_ids=None,
    negative_prompt_attention_mask=None,
):
    processors = LogitsProcessorList()
    if logits_processor is None:
        logits_processor = []

    if generation_config.guidance_scale is not None and generation_config.guidance_scale != 1:
        if getattr(generation_config, "entropy_threshold", None) is None:
            processors.append(
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    generation_config.guidance_scale,
                    self,
                    unconditional_ids=negative_prompt_ids,
                    unconditional_attention_mask=negative_prompt_attention_mask,
                    use_cache=generation_config.use_cache,
                )
            )
        else:
            processors.append(
                ZSGIProcessor(
                    generation_config.guidance_scale,
                    self,
                    entropy_threshold=generation_config.entropy_threshold,
                    tokenizer__=generation_config.tokenizer__,
                    lightweight_negative_prompt=generation_config.lightweight_negative_prompt,
                )
            )
    processors = self._merge_criteria_processor_list(processors, logits_processor)
    return processors


GenerationMixin._sample = _sample
GenerationMixin._get_logits_processor = _get_logits_processor


# ---- Main entry ----

def load_gsm8k(num_samples: int = 100):
    """Load GSM8K dataset (test split)."""
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    problems = []
    for i, item in enumerate(ds):
        if i >= num_samples:
            break
        problems.append(
            {
                "question": item["question"],
                "answer": item["answer"].split("####")[-1].strip().replace(",", ""),
                "solution": item["answer"],
            }
        )
    return problems


def extract_answer(text: str) -> Optional[str]:
    """Extract the final numerical answer from generated text."""
    # Try #### format
    if "####" in text:
        ans = text.split("####")[-1].strip()
        return ans.replace(",", "")
    # Try to find the last number
    import re

    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def run_validation(args):
    """Validation experiment: record CFG signals + step correctness."""
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {"dtype": "auto", "device_map": "auto"}
    if getattr(args, "load_in_4bit", False):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.pop("dtype")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    problems = load_gsm8k(args.num_samples)
    results = []

    for idx, prob in enumerate(problems):
        prompt = (
            "Solve the math problem step by step.\n"
            f"Question: {prob['question']}\n"
            "Answer:"
        )
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        prompt_len = model_inputs.input_ids.shape[1]

        gen_config = GenerationConfig(
            entropy_threshold=args.entropy_threshold,
            lightweight_negative_prompt="OUTPUT ERROR",
            guidance_scale=args.guidance_scale,
            tokenizer__=tokenizer,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

        # Reset global signal buffer before generation
        GLOBAL_SIGNAL_BUFFER.clear()

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                generation_config=gen_config,
            )

        output_ids = generated_ids[0][prompt_len:].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)

        # Retrieve recorded signals (copy out before next generation)
        signals = list(GLOBAL_SIGNAL_BUFFER)

        # Step detection
        full_ids = model_inputs.input_ids[0].tolist() + output_ids
        step_spans = tokenize_step_boundaries(full_ids, tokenizer, prompt_len)

        # Per-step signal aggregation
        step_signals_list = []
        token_sigs = (
            {k: [s[k] for s in signals] for k in signals[0]} if signals else {}
        )
        for start, end in step_spans:
            step_token_idx = list(range(start, min(end, len(signals))))
            step_sigs = aggregate_step_signals(token_sigs, step_token_idx)
            step_instability = compute_step_instability(step_sigs)
            step_signals_list.append(
                {
                    "step_idx": len(step_signals_list),
                    "start_token": start,
                    "end_token": end,
                    "signals": step_sigs,
                    "instability": step_instability,
                }
            )

        pred_answer = extract_answer(output_text)
        correct = pred_answer == prob["answer"]

        results.append(
            {
                "idx": idx,
                "question": prob["question"],
                "gold_answer": prob["answer"],
                "pred_answer": pred_answer,
                "correct": correct,
                "output_text": output_text,
                "num_steps": len(step_spans),
                "step_signals": step_signals_list,
            }
        )

        if (idx + 1) % 10 == 0:
            print(
                f"Processed {idx + 1}/{len(problems)}, accuracy so far: "
                f"{sum(r['correct'] for r in results) / len(results):.2%}"
            )

    # Save results
    out_path = args.output or "validation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Overall accuracy: {sum(r['correct'] for r in results) / len(results):.2%}")


# Global buffer removed — signals now flow through GLOBAL_SIGNAL_BUFFER defined above.


def run_interactive(args):
    """Run a single prompt interactively (for debugging)."""
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_kwargs = {"dtype": "auto", "device_map": "auto"}
    if getattr(args, "load_in_4bit", False):
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.pop("dtype")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    prompt = args.prompt or "What is 12 * 17?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    gen_config = GenerationConfig(
        entropy_threshold=args.entropy_threshold,
        lightweight_negative_prompt="OUTPUT ERROR",
        guidance_scale=args.guidance_scale,
        tokenizer__=tokenizer,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, generation_config=gen_config)

    output_text = tokenizer.decode(
        generated_ids[0][model_inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    print("Generated:", output_text)


def main():
    parser = argparse.ArgumentParser(description="ZSGI HuggingFace prototype")
    parser.add_argument("--model", type=str, required=True, help="Model path or HF id")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["validation", "interactive"],
        default="interactive",
    )
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=100.0)
    parser.add_argument("--entropy_threshold", type=float, default=-1.0)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--load_in_4bit", action="store_true", help="Load model in 4-bit quantization")
    args = parser.parse_args()

    if args.mode == "validation":
        run_validation(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
