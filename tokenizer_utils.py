class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None
    
    def __repr__(self):
        return f"Node({self.value})"
    

def Build_linked_list(s: str, byte_shuffle: list[int] = None) -> list[Node]:
    DLL = []
    ids = list(s.encode('utf-8'))
    for id in ids:
        if byte_shuffle is not None and id < 256:
            id = byte_shuffle[id]
        new_node = Node(id)
        DLL.append(new_node)
    for i in range(len(DLL) - 1):
        DLL[i].next = DLL[i + 1]
        DLL[i + 1].prev = DLL[i]
    return DLL

import tiktoken
enc = tiktoken.get_encoding("gpt2") # this is the GPT-2 tokenizer


def bpe(mergeable_ranks, token, max_rank):
    # helper function used in get_gpt4_merges() to reconstruct the merge forest
    parts = [bytes([b]) for b in token]
    while True:
        min_idx = None
        min_rank = None
        for i, pair in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(pair[0] + pair[1])
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx = i
                min_rank = rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        assert min_idx is not None
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts

def recover_merges(mergeable_ranks):
    # the `merges` are already the byte sequences in their merged state.
    # so we have to recover the original pairings. We can do this by doing
    # a small BPE training run on all the tokens, in their order.
    # also see https://github.com/openai/tiktoken/issues/60
    # also see https://github.com/karpathy/minbpe/issues/11#issuecomment-1950805306
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue # skip raw bytes
        pair = tuple(bpe(mergeable_ranks, token, max_rank=rank))
        assert len(pair) == 2
        # recover the integer ranks of the pair
        ix0 = mergeable_ranks[pair[0]] 
        ix1 = mergeable_ranks[pair[1]]
        merges[(ix0, ix1)] = rank
    return merges

merges = recover_merges(enc._mergeable_ranks)
vocab = {enc._mergeable_ranks[bytes([idx])]: bytes([idx]) for idx in range(256)}
for pair, idx in merges.items():
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

byte_shuffle = {i:enc._mergeable_ranks[bytes([i])] for i in range(256)}