# mySAE

A from-scratch GPT-2 (124M) implementation with a Sparse Autoencoder (SAE) for mechanistic interpretability.

## Project Structure

### GPT-2 Model
| File | Description |
|------|-------------|
| `model.py` | GPT-2 architecture (768 hidden dim, 12 layers, 12 heads) |
| `config.py` | Model and training hyperparameters |
| `run_pretrained.py` | Load checkpoint and run interactive text generation |
| `data_utils.py` | Tokenization and FineWeb-Edu data streaming |
| `regex_tokenizer.py` | Custom BPE tokenizer |
| `tokenizer_utils.py` | GPT-2 tokenizer compatibility |

### Sparse Autoencoder
| File | Description |
|------|-------------|
| `sae_tutorial.md` | Conceptual walkthrough of SAE theory and implementation |
| `sae.py` | SAE model (~65 lines) with config |
| `train_sae.py` | Training script — hooks into frozen GPT-2, trains SAE on activations |
| `eval_sae.py` | Analysis — reconstruction stats, top activating tokens, feature density |

## Quick Start

### Generate text with GPT-2

```bash
python run_pretrained.py --model myGPT2PB.pt
```

### Train the SAE

```bash
# Default: layer 6 residual stream, 8x expansion (6144 features), 10M tokens
python train_sae.py

# Custom config
python train_sae.py --hook_layer 3 --d_sae 12288 --l1_coeff 1e-3 --total_tokens 100000000
```

**Recommended hardware:** CUDA GPU (H100/H200). MPS works but is slow.

Training args:
| Arg | Default | Description |
|-----|---------|-------------|
| `--checkpoint` | `myGPT2PB.pt` | GPT-2 checkpoint |
| `--hook_layer` | `6` | Transformer block to hook (0-11) |
| `--hook_site` | `residual` | `residual`, `mlp`, or `attn` |
| `--d_sae` | `6144` | SAE hidden dimension |
| `--l1_coeff` | `5e-3` | L1 sparsity penalty |
| `--lr` | `3e-4` | Learning rate |
| `--gpt2_batch_size` | `8` | Sequences per GPT-2 forward pass |
| `--total_tokens` | `10000000` | Total tokens to train on |

### Evaluate the SAE

```bash
# Reconstruction quality (MSE, explained variance, L0, dead features)
python eval_sae.py --sae sae_checkpoint.pt --mode stats

# Inspect what a feature responds to
python eval_sae.py --sae sae_checkpoint.pt --mode top_tokens --feature 42

# Feature firing frequency distribution
python eval_sae.py --sae sae_checkpoint.pt --mode density
```

## How It Works

The SAE learns sparse, interpretable features from GPT-2's internal activations:

1. **Hook** into a transformer block using `register_forward_hook`
2. **Collect** activation vectors (one per token position, 768-dimensional)
3. **Encode** into a sparse overcomplete representation (6144-dimensional, mostly zeros)
4. **Decode** back and minimize reconstruction error + sparsity penalty

The core forward pass is 4 lines:
```python
h     = ReLU(W_enc @ (x - b_dec) + b_enc)   # encode: sparse features
x_hat = W_dec @ h + b_dec                    # decode: reconstruct
mse   = (x - x_hat).pow(2).mean()            # reconstruction loss
l1    = h.abs().mean()                        # sparsity loss
```

See `sae_tutorial.md` for a full conceptual walkthrough.

## References

- Bricken et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" (Anthropic, 2023)
- Templeton et al., "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" (Anthropic, 2024)
