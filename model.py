from __future__ import annotations
# (Rerun) Building blocks of nanoGPT
import torch
import torch.nn as nn
# from config import n_emb, device, n_ffd_hidden
import torch.nn.functional as F
import math
import config

class MultiHeadAttention(nn.Module):
    def __init__(self, n_emb, n_heads, dropout): 
        super().__init__()
        assert n_emb % n_heads == 0
        self.n_emb   = n_emb 
        self.num_heads= n_heads
        self.head_size= n_emb // n_heads
        self.dropout_p = dropout

        self.qkv = nn.Linear(n_emb, 3 * n_emb, bias=config.bias)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.c_proj = nn.Linear(n_emb, n_emb, bias=config.bias)

    def forward(self, x):
        #grab sizes
        B, T, C = x.shape

        #slice qkv to q, k, v
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1) # Each has a size of (B, T, C=n_embd)

        #reshape to multiple heads (B, T, C) ---> (B, T, nH, Hs) ---> (B, nH, T, Hs)
        nH, Hs = self.num_heads, self.head_size
        q = q.view(B, T, nH, Hs).transpose(1, 2)
        k = k.view(B, T, nH, Hs).transpose(1, 2)
        v = v.view(B, T, nH, Hs).transpose(1, 2)
        
        #construct the attention matrix and multiply it with the values: h = wei * v
        dropout_p = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p = dropout_p)

        #reshape to (B, T, C) and mix it up
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        
        #dropout will be added at the transformer block level
        return out

    def _forward_with_cache(self, x, kv_cache=None):
        """Forward pass that uses and returns KV cache (used by generate_kv_cache)."""
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        nH, Hs = self.num_heads, self.head_size
        q = q.view(B, T, nH, Hs).transpose(1, 2)
        k = k.view(B, T, nH, Hs).transpose(1, 2)
        v = v.view(B, T, nH, Hs).transpose(1, 2)

        if kv_cache is not None:
            k_cached, v_cached = kv_cache
            k = torch.cat([k_cached, k], dim=2) # concatenation works even if k is not contiguous.
            v = torch.cat([v_cached, v], dim=2)

        new_kv_cache = (k, v)

        # Prefill (no cache): use causal mask for full sequence
        # Decode (with cache, T_q=1): single query attends to all past keys, no mask needed
        is_causal = (kv_cache is None and T > 1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=0.0)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        return out, new_kv_cache

class FeedForward(nn.Module):
    def __init__(self, n_emb, n_ffd_hidden):
        super().__init__()
        self.ffd = nn.Sequential(
            nn.Linear(n_emb, n_ffd_hidden, bias=config.bias),
            nn.GELU(approximate="tanh")  # GPT-2 uses gelu_new (tanh approximation)
        )
        self.c_proj = nn.Linear(n_ffd_hidden, n_emb, bias=config.bias)


    def forward(self, x):
        out = self.ffd(x)
        out = self.c_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, n_emb, n_heads, n_ffd_hidden, dropout):
        super().__init__()
        self.atten = nn.Sequential(
            nn.LayerNorm(n_emb, bias=config.bias),
            MultiHeadAttention(n_emb, n_heads, dropout),
            nn.Dropout(dropout)
        )
        self.ffd = nn.Sequential(
            nn.LayerNorm(n_emb, bias=config.bias),
            FeedForward(n_emb, n_ffd_hidden),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x = x + self.atten(x)
        out = x + self.ffd(x)
        return out

    def _forward_with_cache(self, x, kv_cache=None):
        """Forward pass with KV cache (used by generate_kv_cache)."""
        # Manually unpack nn.Sequential: [0]=LayerNorm, [1]=MHA, [2]=Dropout
        h = self.atten[0](x)                                       # LayerNorm
        h, new_kv_cache = self.atten[1]._forward_with_cache(h, kv_cache)  # MHA
        h = self.atten[2](h)                                       # Dropout (identity at eval)
        x = x + h
        x = x + self.ffd(x)
        return x, new_kv_cache
    
class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: config.Config = config.cfg, verbose: bool = False):
        super().__init__()
        # Single entry point: pass a Config object (default: config.cfg)
        vocab_size = cfg.vocab_size
        n_emb = cfg.n_emb
        n_heads = cfg.n_heads
        n_ffd_hidden = cfg.n_ffd_hidden
        n_layers = cfg.n_layers
        T = cfg.T
        dropout = cfg.dropout
        weight_tying = cfg.weight_tying

        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)
        self.T = T
        self.position_embedding_table = nn.Embedding(T, n_emb)
        self.register_buffer("position_ids", torch.arange(T))  # shape [1, T]
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(n_emb, n_heads, n_ffd_hidden, dropout) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_emb, bias=config.bias)
        self.lm_head = nn.Linear(n_emb, vocab_size, bias=False)
    
        self.apply(self._init_weights)
        if weight_tying:
            # Weight tying: embedding and disembedding should follow the same matrix. Need to turn off bias!
            self.lm_head.weight = self.token_embedding_table.weight
            # Not initialized differently in GPT-2
            # nn.init.normal_(self.token_embedding_table.weight, 0.0, 1.0 / math.sqrt(n_emb))
        # GPT-2 initialization of the final linear layers before residual connection. avoids variance to cumulate.
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))
        
        if verbose:
            print(f"\nThe model has {sum(p.numel() for p in self.parameters())/1e6}M parameters.\n", flush=True)


    def _init_weights(self, module): 
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02) 
            if module.bias is not None: 
                nn.init.zeros_(module.bias) 
        elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(self.position_ids[:T])  # slice up to T
        x = tok_emb + pos_emb # note broadcasting, size is (B, T, C)
        x = self.dropout(x)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_final(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # Flatten logits and targets to (B*T, C) and (B*T,) respectively
            logits = logits.view(B*T, C)
            targets = targets.view(B*T) # targets has a shape of (B, T)
            # Add label smoothing to not preach on sparse samples
            loss = F.cross_entropy(logits, targets, label_smoothing=config.label_smoothing)
            # alternatively, loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    
    def generate(self, idx, max_new_tokens, beta=1.0):
        for _ in range(max_new_tokens):
            # We allow idx to grow in length for output, but only retain the last chunk as
            # context. Last chunk is of context_length or full length, whichever is smaller.
            idx_cond = idx[:, -self.T:]
            logits, loss = self.forward(idx_cond)
            # logits are for every single token, for inference we only need the last one
            logits = logits[:, -1, :]
            probs = F.softmax(logits * beta, dim=-1)
            # use top-k logits (probs) to screen out junk tokens
            top50_probs, top50_indices = torch.topk(probs, 50, dim=-1)
            top50_index = torch.multinomial(top50_probs, num_samples=1)
            idx_next = torch.gather(top50_indices, -1, top50_index)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def _forward_with_cache(self, idx, kv_caches, start_pos):
        """Single forward pass with KV cache (used by generate_kv_cache)."""
        T = idx.shape[1]
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(self.position_ids[start_pos:start_pos + T])
        x = self.dropout(tok_emb + pos_emb)

        for i, block in enumerate(self.blocks):
            x, kv_caches[i] = block._forward_with_cache(x, kv_cache=kv_caches[i])

        x = self.ln_final(x)
        logits = self.lm_head(x[:, -1, :])  # (B, vocab_size)
        return logits, kv_caches

    @torch.no_grad()
    def generate_with_cache(self, idx, max_new_tokens, beta=1.0):
        """Generate with KV cache: O(1) per token instead of O(n) after prefill."""
        n_layers = len(self.blocks)
        kv_caches = [None] * n_layers

        # --- Prefill: process the full prompt, build KV caches ---
        idx_cond = idx[:, -self.T:]
        logits, kv_caches = self._forward_with_cache(idx_cond, kv_caches, start_pos=0)

        probs = F.softmax(logits * beta, dim=-1)
        top50_probs, top50_indices = torch.topk(probs, 50, dim=-1)
        top50_index = torch.multinomial(top50_probs, num_samples=1)
        idx_next = torch.gather(top50_indices, -1, top50_index)
        idx = torch.cat([idx, idx_next], dim=1)

        cur_pos = idx_cond.shape[1]  # position index for the next token to feed

        # --- Decode: one token at a time, reusing cached K/V ---
        for _ in range(max_new_tokens - 1):
            # If we'd exceed the context window, trim the oldest entry from each cache
            if cur_pos >= self.T:
                for i in range(n_layers):
                    k, v = kv_caches[i]
                    kv_caches[i] = (k[:, :, 1:, :], v[:, :, 1:, :])
                cur_pos = self.T - 1

            logits, kv_caches = self._forward_with_cache(idx_next, kv_caches, start_pos=cur_pos)

            probs = F.softmax(logits * beta, dim=-1)
            top50_probs, top50_indices = torch.topk(probs, 50, dim=-1)
            top50_index = torch.multinomial(top50_probs, num_samples=1)
            idx_next = torch.gather(top50_indices, -1, top50_index)
            idx = torch.cat([idx, idx_next], dim=1)
            cur_pos += 1

        return idx

    @classmethod
    def load_gpt2_from_hf(
        cls,
        *,
        hf_model_name: str = "gpt2",
        local_files_only: bool = False,
    ):
        """Load HuggingFace GPT-2 weights into a new GPTLanguageModel.

        Notes
        -----
        - Hyperparameters (vocab_size, n_emb, etc.) are extracted from the HF model config.
        - HuggingFace GPT-2 uses Conv1D modules whose saved weight matrices are
          transposed relative to nn.Linear; this method handles that.
        - Requires the `transformers` package at runtime.
        """
        if not config.bias:
            print("Setting config.bias = True, as in GPT2.")
            config.bias = True

        from transformers import GPT2LMHeadModel

        hf = GPT2LMHeadModel.from_pretrained(hf_model_name, local_files_only=local_files_only)
        hf_config = hf.config

        # Extract hyperparameters from HF config
        vocab_size = hf_config.vocab_size
        n_emb = hf_config.n_embd
        n_heads = hf_config.n_head
        n_layers = hf_config.n_layer
        T = hf_config.n_positions
        dropout = 0.0  # Not in HF config, set to 0 for inference

        # Build a config derived from the HF model
        hf_cfg = config.Config(
            vocab_size=vocab_size,
            n_emb=n_emb,
            n_heads=n_heads,
            n_layers=n_layers,
            T=T,
            dropout=dropout,
            weight_tying=True,
        )

        model = cls(cfg=hf_cfg)

        hf_sd = hf.state_dict()
        ours_sd = model.state_dict()

        def _copy(dst_key: str, src_key: str, *, transpose: bool = False) -> None:
            src = hf_sd[src_key]
            if transpose:
                src = src.t()
            dst = ours_sd[dst_key]
            src = src.to(device=dst.device, dtype=dst.dtype)
            dst.copy_(src)

        with torch.no_grad():
            _copy("token_embedding_table.weight", "transformer.wte.weight")
            _copy("position_embedding_table.weight", "transformer.wpe.weight")
            _copy("ln_final.weight", "transformer.ln_f.weight")
            _copy("ln_final.bias", "transformer.ln_f.bias")

            for i in range(len(model.blocks)):
                _copy(f"blocks.{i}.atten.0.weight", f"transformer.h.{i}.ln_1.weight")
                _copy(f"blocks.{i}.atten.0.bias", f"transformer.h.{i}.ln_1.bias")

                _copy(
                    f"blocks.{i}.atten.1.qkv.weight",
                    f"transformer.h.{i}.attn.c_attn.weight",
                    transpose=True,
                )
                _copy(f"blocks.{i}.atten.1.qkv.bias", f"transformer.h.{i}.attn.c_attn.bias")

                _copy(
                    f"blocks.{i}.atten.1.c_proj.weight",
                    f"transformer.h.{i}.attn.c_proj.weight",
                    transpose=True,
                )
                _copy(f"blocks.{i}.atten.1.c_proj.bias", f"transformer.h.{i}.attn.c_proj.bias")

                _copy(f"blocks.{i}.ffd.0.weight", f"transformer.h.{i}.ln_2.weight")
                _copy(f"blocks.{i}.ffd.0.bias", f"transformer.h.{i}.ln_2.bias")

                _copy(
                    f"blocks.{i}.ffd.1.ffd.0.weight",
                    f"transformer.h.{i}.mlp.c_fc.weight",
                    transpose=True,
                )
                _copy(f"blocks.{i}.ffd.1.ffd.0.bias", f"transformer.h.{i}.mlp.c_fc.bias")

                _copy(
                    f"blocks.{i}.ffd.1.c_proj.weight",
                    f"transformer.h.{i}.mlp.c_proj.weight",
                    transpose=True,
                )
                _copy(f"blocks.{i}.ffd.1.c_proj.bias", f"transformer.h.{i}.mlp.c_proj.bias")

            if model.lm_head.weight.data_ptr() != model.token_embedding_table.weight.data_ptr():
                _copy("lm_head.weight", "lm_head.weight")

        return model
