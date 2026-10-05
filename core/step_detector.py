"""
ZSGI Core: Step Boundary Detection

Detects step boundaries in generated CoT text using heuristics:
  - Newline characters (primary signal in most CoT formats)
  - Sentence-ending punctuation (. ? !)
  - Discourse connectives (因此, 所以, Then, Therefore, etc.)
  - Numbered step markers (1., 2., Step 1:, etc.)

Provides token-level step segmentation given the generated token IDs and
a tokenizer.
"""

import re
from typing import List, Tuple, Optional


# Discourse connectives that often signal a new reasoning step
CONNECTIVES_EN = [
    "then", "therefore", "thus", "hence", "so", "next", "finally",
    "however", "moreover", "furthermore", "additionally", "consequently",
    "because", "since", "as", "now", "first", "second", "third",
]
CONNECTIVES_ZH = [
    "因此", "所以", "于是", "那么", "然后", "接着", "接下来", "最后",
    "但是", "而且", "此外", "另外", "因为", "由于", "既然", "首先",
    "其次", "第一", "第二", "第三",
]

# Step marker patterns: "1.", "2)", "Step 1:", "步骤一：" etc.
STEP_MARKER_PATTERNS = [
    re.compile(r"^\s*\d+\.\s"),           # "1. "
    re.compile(r"^\s*\d+\)\s"),           # "1) "
    re.compile(r"^\s*[Ss]tep\s*\d+[:：.]"),  # "Step 1:"
    re.compile(r"^\s*步骤\s*[一二三四五六七八九十\d]+[:：.]"),
    re.compile(r"^\s*[（(]\s*\d+\s*[)）]\s"),  # "(1) "
]


def detect_step_boundaries_text(text: str) -> List[int]:
    """Return character indices where new steps begin.

    A step boundary is triggered by:
      - A newline followed by non-whitespace (blank-line separated steps)
      - A sentence-ending punctuation followed by a discourse connective
      - A numbered step marker

    Args:
        text: the generated reasoning text

    Returns:
        sorted list of character indices marking step starts (index 0 always)
    """
    boundaries = {0}

    # Newline-based boundaries
    for m in re.finditer(r"\n\s*\n", text):
        boundaries.add(m.end())
    for m in re.finditer(r"\n", text):
        # Single newline also often starts a new step in CoT
        if m.end() < len(text) and not text[m.end()].isspace():
            boundaries.add(m.end())

    # Sentence + connective boundaries
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    pos = 0
    for i, sent in enumerate(sentences):
        if i == 0:
            pos += len(sent)
            continue
        # Check if this sentence starts with a connective
        lower = sent.lstrip().lower()
        starts_connective = any(
            lower.startswith(c) for c in CONNECTIVES_EN
        ) or any(sent.lstrip().startswith(c) for c in CONNECTIVES_ZH)
        if starts_connective:
            boundaries.add(pos)
        # Check for step markers
        if any(p.match(sent) for p in STEP_MARKER_PATTERNS):
            boundaries.add(pos)
        pos += len(sent) + 1  # +1 for the whitespace delimiter

    return sorted(boundaries)


def tokenize_step_boundaries(
    token_ids: List[int],
    tokenizer,
    prompt_length: int,
) -> List[Tuple[int, int]]:
    """Map character-level step boundaries to token-level spans.

    Args:
        token_ids: full generated token ids (prompt + generation)
        tokenizer: the tokenizer used to decode
        prompt_length: number of prompt tokens (boundaries only in generation)

    Returns:
        list of (start_token_idx, end_token_idx) spans for each step,
        where indices are relative to the generated portion (0-indexed).
    """
    gen_ids = token_ids[prompt_length:]
    if len(gen_ids) == 0:
        return []

    text = tokenizer.decode(gen_ids, skip_special_tokens=False)
    char_boundaries = detect_step_boundaries_text(text)

    # Map char boundaries to token boundaries using offset mapping
    # We decode progressively to find which token each char belongs to
    token_char_spans = []  # (start_char, end_char) for each generated token
    cur = 0
    for i in range(len(gen_ids)):
        tok_text = tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=False)
        end = len(tok_text)
        token_char_spans.append((cur, end))
        cur = end

    # Convert char boundaries to token indices
    step_token_starts = []
    for cb in char_boundaries:
        if cb >= len(text):
            continue
        # Find the first token whose char span contains cb
        tidx = 0
        for j, (s, e) in enumerate(token_char_spans):
            if s <= cb < e:
                tidx = j
                break
            if cb >= e:
                tidx = j + 1
        step_token_starts.append(tidx)

    # Deduplicate and build spans
    step_token_starts = sorted(set(step_token_starts))
    if not step_token_starts or step_token_starts[0] != 0:
        step_token_starts = [0] + step_token_starts
    step_token_starts.append(len(gen_ids))

    spans = []
    for i in range(len(step_token_starts) - 1):
        spans.append((step_token_starts[i], step_token_starts[i + 1]))
    return spans


class StepDetector:
    """Stateful step detector that tracks steps during generation."""

    def __init__(self, tokenizer, prompt_length: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.gen_tokens: List[int] = []
        self.step_spans: List[Tuple[int, int]] = []

    def update(self, new_token_id: int):
        """Append a newly generated token and recompute step spans."""
        self.gen_tokens.append(new_token_id)
        full_ids = [0] * self.prompt_length + self.gen_tokens
        self.step_spans = tokenize_step_boundaries(
            full_ids, self.tokenizer, self.prompt_length
        )

    @property
    def num_steps(self) -> int:
        return len(self.step_spans)

    def get_current_step_tokens(self) -> List[int]:
        """Return token indices (relative to generation) for the current step."""
        if not self.step_spans:
            return list(range(len(self.gen_tokens)))
        start, end = self.step_spans[-1]
        return list(range(start, min(end, len(self.gen_tokens))))

    def get_step_tokens(self, step_idx: int) -> Optional[List[int]]:
        if step_idx < 0 or step_idx >= len(self.step_spans):
            return None
        start, end = self.step_spans[step_idx]
        return list(range(start, end))
