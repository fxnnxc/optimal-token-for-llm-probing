from typing import Dict, List, Tuple, Optional, Union
import random

# -----------------------------
# Configuration / Defaults
# -----------------------------
DEFAULT_RANDOM_WORDS = ["random", "token", "probe", "alpha", "beta"]
DEFAULT_RANDOM_SENTENCES = [
    "Random text is inserted.",
    "Noise sequence without semantics.",
    "Arbitrary tokens appended here."
]
DEFAULT_SYNTAX_SENTENCES_VARIABLE = [
    "Think step by step.",
    "Think as a careful analyst.",
    "Think through it slowly."
]
FIXED_SYNTAX_SENTENCE = "Think step by step."  # For [Fixed / Syntactical / Multi]

# -----------------------------
# Helper functions
# -----------------------------

def _ensure_endswith_punct(s: str, punct: str) -> str:
    s = s.rstrip()
    if not s.endswith(punct):
        s = s + punct
    return s

def _append_with_space(base: str, suffix: str) -> str:
    base = base.rstrip()
    # Add a leading space if needed to avoid merging words
    if base and not base.endswith((" ", "\n", "\t")):
        return base + " " + suffix
    return base + suffix

def _find_tail_span(hay: List[int], needle: List[int]) -> Optional[Tuple[int, int]]:
    """
    Find the *last* occurrence of 'needle' (token IDs) as a contiguous subsequence in 'hay'.
    Returns (start_idx, end_idx_exclusive) or None if not found.
    """
    if not needle or len(needle) > len(hay):
        return None
    # Search from the tail
    last_start = None
    n = len(needle)
    for start in range(len(hay) - n, -1, -1):
        if hay[start:start + n] == needle:
            last_start = start
            break
    if last_start is None:
        return None
    return (last_start, last_start + n)

def _decode_tokens(tokenizer, input_ids: List[int]) -> List[str]:
    # Best-effort piecewise decoding (works for fast tokenizers)
    # If you prefer, you can just return tokenizer.convert_ids_to_tokens(input_ids)
    try:
        return tokenizer.convert_ids_to_tokens(input_ids)
    except Exception:
        return [tokenizer.decode([tid], skip_special_tokens=False) for tid in input_ids]

# -----------------------------
# Main API
# -----------------------------

def prepare_probe_inputs(
    sample_text: str,
    fixedness: str,     # "fixed" | "variable"
    semantics: str,     # "syntactical" | "special" | "random"
    count: str,         # "single" | "multi"
    tokenizer,
    use_chat_template: bool = True,
    random_seed: int = 7,
    assume_anchor_neg6: bool = True
) -> Dict[str, Union[str, List[int], List[str], List[int]]]:
    """
    Build the final (chat-formatted) text, tokenize it, and compute probe positions.

    Returns:
        {
          "final_text": str,              # the final user text used in messages (pre-template)
          "chat_formatted_text": str,     # printable chat-formatted text (if available; else same as final_text)
          "input_ids": List[int],         # tokenized ids of the fully formatted prompt
          "tokens": List[str],            # token pieces
          "probe_positions": List[int],   # indices at which to extract hidden states
        }
    """
    rng = random.Random(random_seed)

    fixedness = fixedness.lower().strip()
    semantics = semantics.lower().strip()
    count     = count.lower().strip()
    assert fixedness in {"fixed", "variable"}
    assert semantics in {"syntactical", "special", "random"}
    assert count in {"single", "multi"}

    # 1) Build the *user* text according to condition
    user_text = sample_text

    # A/B/C mapping based on your 12 cases
    if fixedness == "fixed" and semantics == "syntactical" and count == "single":
        # End with '.' and probe at -6
        user_text = _ensure_endswith_punct(user_text, ".")
        suffix_for_span = None  # No need to find span; single-token at -6

    elif fixedness == "fixed" and semantics == "special" and count == "single":
        # No user text change; probe at -5 (special). (Template-resolved)
        suffix_for_span = None

    elif fixedness == "fixed" and semantics == "random" and count == "single":
        # Append a fixed random word and probe at -6
        word = DEFAULT_RANDOM_WORDS[0]
        user_text = _append_with_space(user_text, word)
        suffix_for_span = None

    elif fixedness == "fixed" and semantics == "syntactical" and count == "multi":
        # Append a fixed syntactical phrase and collect its span
        user_text = _append_with_space(user_text, FIXED_SYNTAX_SENTENCE)
        suffix_for_span = FIXED_SYNTAX_SENTENCE

    elif fixedness == "fixed" and semantics == "special" and count == "multi":
        # No text change; use special tokens at (-5..-1)
        suffix_for_span = None

    elif fixedness == "fixed" and semantics == "random" and count == "multi":
        # Append a fixed random sentence and collect its span
        sent = DEFAULT_RANDOM_SENTENCES[0]
        user_text = _append_with_space(user_text, sent)
        suffix_for_span = sent

    elif fixedness == "variable" and semantics == "syntactical" and count == "single":
        # End with '.' or '?' (variable), probe at -6
        punct = rng.choice([".", ",", "!"])
        user_text = user_text.rstrip()
        if user_text.endswith("."):
            user_text = user_text[:-1]
        user_text = _ensure_endswith_punct(user_text, punct)
        suffix_for_span = None

    elif fixedness == "variable" and semantics == "special" and count == "single":
        # Not typical; we keep text unchanged. You may decide to raise if you prefer.
        
        suffix_for_span = None

    elif fixedness == "variable" and semantics == "random" and count == "single":
        # Append variable random word(s), probe at -6
        word = rng.choice(DEFAULT_RANDOM_WORDS)
        user_text = _append_with_space(user_text, word)
        suffix_for_span = None

    elif fixedness == "variable" and semantics == "syntactical" and count == "multi":
        # Append one phrase chosen from a set; collect its span (< -6)
        phrase = rng.choice(DEFAULT_SYNTAX_SENTENCES_VARIABLE)
        user_text = _append_with_space(user_text, phrase)
        suffix_for_span = phrase

    elif fixedness == "variable" and semantics == "special" and count == "multi":
        # Not typical; no change, but you could raise if needed.
        suffix_for_span = None

    elif fixedness == "variable" and semantics == "random" and count == "multi":
        # Append variable random sentence; collect its span (< -6)
        sent = rng.choice(DEFAULT_RANDOM_SENTENCES)
        user_text = _append_with_space(user_text, sent)
        suffix_for_span = sent

    # 2) Build chat-formatted input (messages → template → input_ids)
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": user_text}]
        # tokenize=True returns input_ids; return_tensors=None keeps them as a list
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True
        )
        input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded
        chat_formatted_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        # Fallback: simple raw encode
        chat_formatted_text = user_text
        input_ids = tokenizer.encode(
            user_text,
            add_special_tokens=True
        )

    # 3) Compute probe positions
    probe_positions: List[int] = []

    # 3a) "special" cases: (-5..-1) or -5
    if semantics == "special":
        if count == "single":
            # -5
            probe_positions = [len(input_ids) - 5]
        else:
            # multi: (-5, -4, -3, -2, -1)
            n = len(input_ids)
            probe_positions = [n - 5, n - 4, n - 3, n - 2, n - 1]

    # 3b) "syntactical" or "random" with *single*: anchor at -6
    elif count == "single":
        if assume_anchor_neg6:
            probe_positions = [len(input_ids) - 6]
        else:
            # If you don’t want to assume -6, you can implement your own anchor finder.
            raise ValueError("Anchor (-6) not assumed; provide your own anchor resolver.")

    # 3c) "syntactical" or "random" with *multi*: locate the appended suffix span
    else:
        if suffix_for_span is None:
            # No suffix span to find; you might choose to anchor to (-6) or raise
            if assume_anchor_neg6:
                probe_positions = [len(input_ids) - 6]  # fallback
            else:
                raise ValueError("Multi requires suffix span or anchor, but none was provided.")
        else:
            # Tokenize the suffix alone to find tail span
            suffix_ids = tokenizer.encode(
                suffix_for_span,
                add_special_tokens=False
            )
            span = _find_tail_span(input_ids, suffix_ids)
            if span is None:
                # If not found (e.g., whitespace/tokenization mismatch), fallback to < -6 window
                if assume_anchor_neg6:
                    # Back off to a small window prior to -6
                    anchor = len(input_ids) - 5
                    # take a safe window of same length as suffix or up to 8 tokens
                    win = min(len(suffix_ids) if suffix_ids else 1, 5)
                    probe_positions = list(range(max(0, anchor - win), anchor))
                else:
                    raise RuntimeError("Could not locate suffix span in tokenized sequence.")
            else:
                start, end = span
                probe_positions = list(range(start, end))

    # 4) Pack results
    tokens = _decode_tokens(tokenizer, input_ids)

    return {
        "final_text": user_text,                 # the user content used to build messages
        "chat_formatted_text": chat_formatted_text,  # printable form of the full prompt
        "input_ids": input_ids,
        "decoded_tokens": tokens,
        "probe_positions": probe_positions,
    }

# -----------------------------
# Example usage (pseudo)
# -----------------------------
from typing import Dict, List, Any, Iterable
from torch.utils.data import DataLoader
import torch

def _collate_with_probe_mask(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Collate function that:
      - Pads input_ids to the max length in the batch (LEFT PADDING)
      - Builds attention_mask
      - Builds probe_mask (bool), True at probe positions per sample
      - Also returns (optional) per-sample lengths for convenience
    """
    # lengths
    lengths = [len(ex["input_ids"]) for ex in batch]
    max_len = max(lengths)

    # allocate tensors
    input_ids = torch.full((len(batch), max_len), fill_value=pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    probe_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    sample_ids = torch.tensor([ex['sample_ids'] for ex in batch], dtype=torch.long)
    labels = torch.tensor([ex['label'] for ex in batch], dtype=torch.long)
    # Collect all probe positions for each sample (will be padded to max_len)
    all_probe_positions = []
    
    for i, ex in enumerate(batch):
        ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        L = ids.shape[0]
        
        # LEFT PADDING: place sequence at the end (right side) of the tensor
        pad_start = max_len - L
        input_ids[i, pad_start:] = ids
        attention_mask[i, pad_start:] = 1

        # Adjust probe positions for left padding offset
        adjusted_probe_positions = []
        for p in ex.get("probe_positions", []):
            if 0 <= p < L:
                # Add the padding offset to get the new position
                adjusted_pos = pad_start + p
                probe_mask[i, adjusted_pos] = True
                adjusted_probe_positions.append(adjusted_pos)
        
        # Pad probe positions to max_len with -1 (invalid position)
        padded_probe_pos = []
        for j, pos in enumerate(adjusted_probe_positions):
            if j < max_len:
                padded_probe_pos.append(pos)
        all_probe_positions.append(padded_probe_pos)

    probe_positions = torch.tensor(all_probe_positions, dtype=torch.long)

    return {
        "input_ids": input_ids,            # (B, T)
        "attention_mask": attention_mask,  # (B, T)
        "probe_mask": probe_mask,          # (B, T) bool
        "probe_positions": probe_positions,  # (B, T) int
        "sample_ids": sample_ids,          # (B,)
        "seq_lens": torch.tensor(lengths, dtype=torch.long),  # (B,)
        "labels": labels,          # (B,)
    }

def formatting_prompts(
    dataset: Iterable[Dict[str, Any]],
    fixedness: str,
    semantics: str,
    count: str,
    tokenizer,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    seed=42,
):
    """
    Preprocess a dataset of samples into tokenized, probe-ready batches.

    Input dataset: iterable of dicts with at least {"text": <str>}
    Side-effects per sample:
      - Replaces "text" with the finalized user text actually used
      - Adds "input_ids" (list[int])
      - Adds "probe_positions" (list[int])

    Returns:
      torch.utils.data.DataLoader that yields dict with:
        - input_ids:     (B, T) padded LongTensor
        - attention_mask:(B, T) LongTensor
        - probe_mask:    (B, T) BoolTensor, True where to extract hidden states
        - seq_lens:      (B,)   LongTensor, original sequence lengths
    """
    # Ensure pad token exists
    if tokenizer.pad_token_id is None:
        # set pad token id to eos if missing (common for some LLMs)
        if getattr(tokenizer, "eos_token_id", None) is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; set one before batching.")
        tokenizer.pad_token = tokenizer.eos_token

    processed: List[Dict[str, Any]] = []
    
    rng = random.Random(seed)
    
    if fixedness == "variable" and semantics == "special":
        print("Warning: Variable Special is not supported and equivalent to Fixed Special as Candidates for Special Tokens are Non-trivial")
    
    for sample in dataset:
        sample_text = sample["text"]
        sample_id = sample["sample_id"]

        out = prepare_probe_inputs(
            sample_text=sample_text,
            fixedness=fixedness,
            semantics=semantics,
            count=count,
            tokenizer=tokenizer,
            use_chat_template=True,
            random_seed=rng.randint(0, 1000000),
        )

        processed.append({
            # keep anything else the original sample had
            **sample,
            "text": out["final_text"],
            "input_ids": out["input_ids"],
            "probe_positions": out["probe_positions"],
            "sample_ids": sample_id,
            "labels": sample["label"],
        })

    # Simple list-backed dataset (so DataLoader can index it)
    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, data: List[Dict[str, Any]]):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    ds = _ListDataset(processed)
    collate_fn = lambda batch: _collate_with_probe_mask(batch, tokenizer.pad_token_id)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
