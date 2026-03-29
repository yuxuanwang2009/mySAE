"""Regex-based tokenizer that splits text with a GPT-4-style pattern, trains BPE merges,
supports encode/decode, and can save/load its learned state.

Quick usage:
    tok = RegexTokenizer.train(corpus_text, vocab_size=10_000, path="tokenizer.json")
    ids = tok.encode("hello world")
    text = tok.decode(ids)

    # later/restart
    tok = RegexTokenizer.load("tokenizer.json")
"""

import regex as re
import json
import heapq
from tokenizer_utils import Build_linked_list


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer:
    def __init__(self, merges=None, vocab=None, pattern=None, special_token_to_id=None, byte_shuffle=None):
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.merges = {} if merges is None else merges
        self.vocab = {i: bytes([i]) for i in range(256)} if vocab is None else vocab
        self.special_token_to_id = {} if special_token_to_id is None else special_token_to_id
        self.byte_shuffle = byte_shuffle

    def _merge(self, ids, pair, new_token):
        """
        Replace all non-overlapping occurrences of pair with new_token.
        Needed for training, but not efficient for encoding large texts.
        We will use a min-heap based priority queue approach for efficient encoding.
        """
        i = 0
        out_ids = []
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out_ids.append(new_token)
                i += 2
            else:
                out_ids.append(ids[i])
                i += 1
        return out_ids

    @classmethod
    def train(cls, text, vocab_size, path = None, special_tokens: list[str] = None, verbose=False):
        if special_tokens is None:
            special_tokens = []

        assert vocab_size >= 256 + len(special_tokens)
        num_merges = vocab_size - 256 - len(special_tokens)

        tokenizer = cls()
        chunks = re.findall(tokenizer.compiled_pattern, text) # chunks is a list of strings
        chunk_ids = [list(chunk.encode("utf-8")) for chunk in chunks] # list converts bytes to a list of ints

        for step in range(num_merges):
            counts = {}
            for ids in chunk_ids:
                for a, b in zip(ids, ids[1:]):
                    pair = (a, b)
                    counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            next_pair = max(counts, key=counts.get)
            new_token = 256 + step
            tokenizer.merges[next_pair] = new_token
            tokenizer.vocab[new_token] = tokenizer.vocab[next_pair[0]] + tokenizer.vocab[next_pair[1]]
            if verbose:
                render_token = None
                translation_table = str.maketrans({" ": "<sp>", "\n": "<nl>"})
                def render_token(token_ids):
                    return tokenizer.decode(token_ids).translate(translation_table)

                left = render_token(next_pair[:1])
                right = render_token(next_pair[1:])
                merged = render_token([new_token])
                count = counts[next_pair]
                print(f"merge {step+1:4d}/{num_merges:<4d} : {left:<10} {right:<10} -> {merged:<12} | {count:>8} occurrences")
            chunk_ids = [tokenizer._merge(ids, next_pair, new_token) for ids in chunk_ids]
        # Reserve the highest token IDs for special tokens
        next_token_id = 256 + num_merges
        for tok in special_tokens:
            tokenizer.special_token_to_id[tok] = next_token_id
            tokenizer.vocab[next_token_id] = tok.encode("utf-8")
            next_token_id += 1

        if path is not None:
            tokenizer.save(path) # for future loading
        return tokenizer # ready to use after training

    def _encode_chunk(self, chunk, byte_shuffle=None):
        ids = list(chunk.encode("utf-8"))
        if byte_shuffle is not None:
            for i, id_val in enumerate(ids):
                if id_val < 256:
                    ids[i] = byte_shuffle[id_val]
        for pair, new_token in self.merges.items():
            ids = self._merge(ids, pair, new_token)
        return ids

    def _encode_chunk_fast(self, chunk, byte_shuffle=None):
        # Before we begin, create a double-linked list (DLL) from the chunk, each element of the list is a Node object with prev, next, value attributes.
        dll = Build_linked_list(chunk, byte_shuffle)
        
        # First, create a priority list (min-heap) by scanning the DLL:
        heap = []
        for node_a in dll[:-1]:
            node_b = node_a.next
            if (node_a.value, node_b.value) in self.merges:
                new_id = self.merges[(node_a.value, node_b.value)]
                heapq.heappush(heap, (new_id, id(node_a), node_a, node_b))
        
        # Next, update the dll and min-heap until heap is empty
        while heap:
            new_id, _, node_a, node_b = heapq.heappop(heap)
            if node_a.next != node_b or node_b.prev != node_a:
                continue  # stale entry if no longer adjacent
            if self.merges.get((node_a.value, node_b.value), None) != new_id:
                continue  # stale entry even if the node values changed
            # Merge node_a and node_b
            node_a.value = new_id
            node_a.next = node_b.next
            if node_b.next is not None:
                node_b.next.prev = node_a
            node_b.prev = None # detach node_b
            node_b.next = None # detach node_b
            # Check for new merge opportunities around the new node_a
            if node_a.next is not None:
                if (node_a.value, node_a.next.value) in self.merges:
                    new_id = self.merges[(node_a.value, node_a.next.value)]
                    heapq.heappush(heap, (new_id, id(node_a), node_a, node_a.next))
            if node_a.prev is not None:
                if (node_a.prev.value, node_a.value) in self.merges:
                    new_id = self.merges[(node_a.prev.value, node_a.value)]
                    heapq.heappush(heap, (new_id, id(node_a.prev), node_a.prev, node_a))
        # Finally, we extract the IDs from the modified DLL:
        ids = [dll[0].value]
        for node in dll[:-1]:
            if node.next is not None:
                ids.append(node.next.value)
        return ids

    def _split_with_special(self, text, allowed_special):
        """
        Split text into chunks using the tokenizer regex, but preserve any
        allowed special tokens as standalone chunks so they are not broken up.
        """
        if not allowed_special:
            return self.compiled_pattern.findall(text)

        # Normalize input: sets/lists are fine; strings should be treated as one token
        if isinstance(allowed_special, str):
            allowed = {allowed_special}
        else:
            allowed = set(allowed_special)

        special_regex = re.compile("|".join(re.escape(tok) for tok in sorted(allowed, key=len, reverse=True)))
        chunks = []
        cursor = 0
        for match in special_regex.finditer(text):
            if match.start() > cursor:
                chunks.extend(self.compiled_pattern.findall(text[cursor:match.start()]))
            chunks.append(match.group(0))
            cursor = match.end()
        if cursor < len(text):
            chunks.extend(self.compiled_pattern.findall(text[cursor:]))
        return [c for c in chunks if c]

    def encode(self, text, allowed_special=None, fast=True):
        chunks = self._split_with_special(text, allowed_special)
        ids = []
        for chunk in chunks:
            if chunk in self.special_token_to_id:
                ids.append(self.special_token_to_id[chunk])
                continue
            if fast:
                ids.extend(self._encode_chunk_fast(chunk, self.byte_shuffle))
            else:
                ids.extend(self._encode_chunk(chunk, self.byte_shuffle))
        return ids

    def decode(self, ids):
        if self.special_token_to_id:
            self.vocab.update({self.special_token_to_id[token]: token.encode("utf-8") for token in self.special_token_to_id})
        text_bytes = b"".join(self.vocab[i] for i in ids)
        return text_bytes.decode("utf-8", errors="replace")

    def save(self, path):
        merges_serialized = [(a, b, tok) for (a, b), tok in self.merges.items()]
        state = {
            "pattern": self.pattern,
            "merges": merges_serialized,
            "special_tokens": list(self.special_token_to_id.items()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        special_token_to_id = dict(state.get("special_tokens", []))
        for a, b, tok in state["merges"]:
            merges[(a, b)] = tok
            vocab[tok] = vocab[a] + vocab[b]
        for tok_str, tok_id in special_token_to_id.items():
            vocab[tok_id] = tok_str.encode("utf-8")
        return cls(
            merges=merges,
            vocab=vocab,
            pattern=state.get("pattern", GPT4_SPLIT_PATTERN),
            special_token_to_id=special_token_to_id,
        )
