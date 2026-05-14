"""
Flexible tokenizer with byte-level encoding support.

Two modes based on config:
1. tiktoken mode (base_tokenizer="cl100k_base"):
   - Token layout: [base vocab | 256 byte tokens | special tokens]
   - |word| syntax to spell out words as bytes

2. HuggingFace mode (base_tokenizer="HuggingFaceTB/SmolLM2-135M"):
   - Token layout: [HF vocab | special tokens added via add_tokens]
   - Byte-level BPE handles all characters natively
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

import torch

SPELL_PATTERN = re.compile(r"(?<!<)\|([^|<>]+)\|(?!>)")


def spell(word: str) -> str:
    return f"|{word}|"


def spell_randomly(text: str, prob: float = 0.2) -> str:
    """Randomly wrap ~prob fraction of words in |pipes| for byte-level spelling."""
    if prob <= 0:
        return text
    result = []
    i = 0
    in_bracket = False
    while i < len(text):
        if text[i] == "[":
            in_bracket = True
            result.append(text[i])
            i += 1
        elif text[i] == "]":
            in_bracket = False
            result.append(text[i])
            i += 1
        elif text[i].isalpha() and not in_bracket:
            word_start = i
            while i < len(text) and text[i].isalpha():
                i += 1
            word = text[word_start:i]
            if len(word) > 1 and random.random() < prob:
                result.append(f"|{word}|")
            else:
                result.append(word)
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


SPECIAL_TOKENS = [
    "[pause]",
    "[hesitate]",
    "[breath]",
    "[env]",
    "[tut]",
    "[mouthnoise]",
    "[sigh]",
    "[scoff]",
    "[gasp]",
    "[sniff]",
    "[music]",
    "[sing]",
    "[laugh]",
    "[cough]",
    "[clearsthroat]",
    "[noise]",
    "[overlap]",
    "[SC]",
    "\U0001F3B5",  # 🎵
    "[shortpause]",
    "[longpause]",
]


TIME_TOKENS = [f"<|{v:.2f}|>" for v in [round(x * 0.02, 2) for x in range(9001)]]


@dataclass
class TokenizerConfig:
    base_tokenizer: str = "gpt2"  # tiktoken name or HF model ID
    add_time_tokens: bool = False


class VuiTokenizer:
    NUM_BYTES = 256

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        self._is_hf = "/" in self.config.base_tokenizer

        self.special_tokens = list(SPECIAL_TOKENS)
        if self.config.add_time_tokens:
            self.special_tokens = self.special_tokens + TIME_TOKENS

        if self._is_hf:
            self._init_hf()
        else:
            self._init_tiktoken()

    def _init_tiktoken(self):
        import tiktoken

        self.base_tokenizer = tiktoken.get_encoding(self.config.base_tokenizer)
        self._base_vocab_size = self.base_tokenizer.n_vocab
        self.byte_offset = self._base_vocab_size
        self.special_offset = self._base_vocab_size + self.NUM_BYTES
        self.special_to_id = {
            tok: self.special_offset + i for i, tok in enumerate(self.special_tokens)
        }
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.pad_token_id = 0
        self.eos_token_id = self.base_tokenizer.eot_token
        self.bos_token_id = None

    def _init_hf(self):
        self._hf_tok = self._load_hf_tokenizer()
        self._base_vocab_size = self._hf_tok.get_vocab_size()
        from tokenizers import AddedToken

        byte_tokens = [AddedToken(f"<|byte_{i}|>") for i in range(self.NUM_BYTES)]
        self.byte_offset = self._base_vocab_size
        self._hf_tok.add_tokens(byte_tokens)
        self.special_offset = self._hf_tok.get_vocab_size()
        special = [AddedToken(t, special=True) for t in self.special_tokens]
        self._hf_tok.add_special_tokens(special)
        self.special_to_id = {
            tok: self._hf_tok.token_to_id(tok) for tok in self.special_tokens
        }
        self.id_to_special = {v: k for k, v in self.special_to_id.items()}
        self.pad_token_id = 0
        self.eos_token_id = self._hf_tok.token_to_id("</s>") or 0
        self.bos_token_id = self._hf_tok.token_to_id("<s>")

    def _load_hf_tokenizer(self):
        import os

        from tokenizers import Tokenizer

        try:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(
                self.config.base_tokenizer, local_files_only=True
            )
        except Exception:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(self.config.base_tokenizer)
        tok_json = os.path.join(model_dir, "tokenizer.json")
        return Tokenizer.from_file(tok_json)

    @property
    def vocab_size(self) -> int:
        if self._is_hf:
            return self._hf_tok.get_vocab_size()
        return self._base_vocab_size + self.NUM_BYTES + len(self.special_tokens)

    def _text_to_bytes(self, text: str) -> list[int]:
        return [b + self.byte_offset for b in text.encode("utf-8")]

    def _is_byte_token(self, tid: int) -> bool:
        return self.byte_offset <= tid < self.byte_offset + self.NUM_BYTES

    def _bytes_to_text(self, byte_ids: list[int]) -> str:
        return bytes(b - self.byte_offset for b in byte_ids).decode(
            "utf-8", errors="replace"
        )

    def _encode_with_base(self, text: str) -> list[int]:
        if self._is_hf:
            return self._hf_tok.encode(text, add_special_tokens=False).ids
        return self.base_tokenizer.encode(text, allowed_special="all")

    def _find_special_tokens(self, text: str) -> list[tuple[int, int, str]]:
        found = []
        for token in self.special_tokens:
            start = 0
            while True:
                idx = text.find(token, start)
                if idx == -1:
                    break
                found.append((idx, idx + len(token), token))
                start = idx + 1

        found.sort(key=lambda x: (x[0], -(x[1] - x[0])))

        non_overlapping = []
        last_end = -1
        for start, end, token in found:
            if start >= last_end:
                non_overlapping.append((start, end, token))
                last_end = end

        return non_overlapping

    def _encode_segment(self, text: str, force_bytes: bool = False) -> list[int]:
        if force_bytes or not text:
            return self._text_to_bytes(text)
        if self._is_hf:
            return self._encode_with_base(text)
        # tiktoken: fall back to byte tokens for tokens with partial UTF-8
        result = []
        for tid in self._encode_with_base(text):
            raw = self.base_tokenizer.decode_single_token_bytes(tid)
            try:
                raw.decode("utf-8")
                result.append(tid)
            except UnicodeDecodeError:
                result.extend(b + self.byte_offset for b in raw)
        return result

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> torch.Tensor:

        tokens = []
        spell_matches = list(SPELL_PATTERN.finditer(text))
        special_positions = self._find_special_tokens(text)

        all_markers: list[tuple[int, int, str, str]] = []
        for m in spell_matches:
            all_markers.append((m.start(), m.end(), m.group(1), "spell"))
        for start, end, tok in special_positions:
            in_spell = any(m.start() <= start < m.end() for m in spell_matches)
            if not in_spell:
                all_markers.append((start, end, tok, "special"))

        all_markers.sort(key=lambda x: x[0])

        pos = 0
        for start, end, content, marker_type in all_markers:
            if start > pos:
                segment = text[pos:start]
                tokens.extend(self._encode_segment(segment, force_bytes=False))

            if marker_type == "spell":
                tokens.extend(self._text_to_bytes(content))
            else:
                tokens.append(self.special_to_id[content])

            pos = end

        if pos < len(text):
            tokens.extend(self._encode_segment(text[pos:], force_bytes=False))

        return torch.tensor(tokens, dtype=torch.long)

    def decode(
        self, token_ids: list[int] | torch.Tensor, preserve_byte_markers: bool = True
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        result = []
        byte_buffer = []

        def flush_bytes():
            if byte_buffer:
                text = self._bytes_to_text(byte_buffer)
                if preserve_byte_markers:
                    result.append(f"|{text}|")
                else:
                    result.append(text)
                byte_buffer.clear()

        for tid in token_ids:
            if tid in self.id_to_special:
                flush_bytes()
                result.append(self.id_to_special[tid])
            elif self._is_byte_token(tid):
                byte_buffer.append(tid)
            else:
                flush_bytes()
                if self._is_hf:
                    result.append(self._hf_tok.decode([tid], skip_special_tokens=False))
                else:
                    try:
                        result.append(self.base_tokenizer.decode([tid]))
                    except Exception:
                        pass

        flush_bytes()
        return "".join(result)

    def __call__(
        self,
        text: str | list[str],
        padding: str = "longest",
        return_tensors: str = "pt",
    ) -> dict:
        if isinstance(text, str):
            text = [text]

        encoded = [self.encode(t) for t in text]

        if padding == "longest":
            max_len = max(len(e) for e in encoded)
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0

            padded = []
            masks = []
            for e in encoded:
                pad_len = max_len - len(e)
                padded.append(
                    torch.cat([e, torch.full((pad_len,), pad_id, dtype=torch.long)])
                )
                mask = torch.cat([torch.ones(len(e)), torch.zeros(pad_len)])
                masks.append(mask)

            input_ids = torch.stack(padded)
            attention_mask = torch.stack(masks)
        else:
            input_ids = torch.nn.utils.rnn.pad_sequence(encoded, batch_first=True)
            attention_mask = (input_ids != 0).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


_default_tokenizer: VuiTokenizer | None = None


def get_tokenizer(config: TokenizerConfig | None = None) -> VuiTokenizer:
    global _default_tokenizer
    if _default_tokenizer is None or config is not None:
        _default_tokenizer = VuiTokenizer(config)
    return _default_tokenizer


if __name__ == "__main__":
    tok = VuiTokenizer()

    print(f"Vocab size: {tok.vocab_size}")
    print(f"Base vocab: 0-{tok._base_vocab_size - 1}")
    print(f"Byte tokens: {tok.byte_offset}-{tok.special_offset - 1}")
    print(
        f"Special tokens: {tok.special_offset}-{tok.special_offset + len(tok.special_tokens) - 1}"
    )
    print()

    test_cases = [
        "Hello world",
        "Hello [pause] world",
        "Say |hello| to me",
        "The word |pneumonoultramicroscopicsilicovolcanoconiosis| is long",
        "[0] <|0.00|>Hello there<|1.50|>",
        "Multiple |spelled| words |here| in text",
    ]

    for text in test_cases:
        print(f"Input: {text}")
        tokens = tok.encode(text)
        print(f"  Tokens ({len(tokens)}): {tokens[:20].tolist()}...")
        print(f"  Decoded: {tok.decode(tokens)}")
        print()

    print("Batch encoding test:")
    batch = tok(
        ["Hello world", "Goodbye [pause] world"],
        padding="longest",
    )
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
