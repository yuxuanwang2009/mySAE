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
| `interpret.py` | Automated interpretability — uses Claude to describe and validate features |
| `steer.py` | Feature steering — amplify/suppress SAE features during generation |

## Setup

```bash
conda create -n mySAE python=3.11 -y && conda activate mySAE
pip install torch datasets tiktoken anthropic
```

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

### Interpret features with Claude

```bash
# Ask Claude to describe what a feature detects, then validate on held-out examples
python interpret.py --sae sae_checkpoint.pt --feature 42

# More data for better accuracy
python interpret.py --sae sae_checkpoint.pt --feature 42 --n_batches 100
```

Requires `ANTHROPIC_API_KEY` set in your environment.

### Steer generation with SAE features

```bash
# Amplify a feature during generation
python steer.py --sae sae_checkpoint.pt --feature 42 --scale 5.0 --prompt "The scientist discovered"

# Suppress a feature
python steer.py --sae sae_checkpoint.pt --feature 42 --scale -5.0 --prompt "The scientist discovered"
```

Generates text with and without steering for comparison.

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

## Case Study: Feature 20652 — Destruction and Loss

### Training the SAE

We trained a 32x expansion SAE with TopK activation on the residual stream of layer 6 of our GPT-2 124M model, using an NVIDIA B200 GPU on HiPerGator:

```bash
python train_sae.py --d_sae 24576 --total_tokens 100000000 --gpt2_batch_size 64 \
    --warmup_steps 100 --activation topk --k 50 --save_path sae_32x_topk.pt
```

Key hyperparameters:
- **Dictionary size:** 24,576 (32x expansion over 768-dim residual stream)
- **Activation:** TopK with k=50 (exactly 50 features active per token)
- **Training data:** 100M tokens from FineWeb-Edu
- **Final explained variance:** 0.83

### Identifying the Feature

Rather than scanning the training corpus for specific tokens, we used **contrastive activation directions** to find features aligned with negative sentiment. We crafted 18 pairs of matched prompts — one negative ("The situation was terrible and getting worse") and one positive ("The situation was wonderful and getting better") — ran them through GPT-2, collected the activations at the target adjective positions, and computed:

```
concept_direction = mean(negative_activations) - mean(positive_activations)
```

We then ranked all 24,576 SAE decoder vectors by cosine similarity to this direction. Feature 20652 was among the top hits.

See `find_feature_by_direction.py` for the full implementation.

### Interpreting the Feature

Using `interpret.py`, we collected the top activating examples for feature 20652 across the training corpus and asked Claude to describe the pattern:

> This feature captures words and phrases describing destruction, elimination, or loss — particularly verbs in the passive voice or past participle forms that indicate something has been damaged, removed, or ceased to exist (sunk, burned, destroyed, killed, lost, wiped out, etc.).

Top activating examples:
| Activation | Token | Context |
|-----------|-------|---------|
| 9.07 | sunk | "...1864 when she was **sunk** by the USS Kears..." |
| 8.72 | burned | "...the shelter in Philadelphia was **burned** by a white mob in..." |
| 8.46 | destroyed | "...His drum was **destroyed** by a shell fragment in..." |
| 8.31 | beheaded | "...the Jew was **beheaded** upon Muaz..." |
| 8.30 | lost | "...to $110 million in **lost** revenue and sales..." |

Claude's validation accuracy on held-out examples: **80%** (16/20 correct).

### Steering with the Feature

We steer generation by adding the feature's decoder direction (a 768-dim unit vector) directly to the residual stream at the last token position during autoregressive generation:

```python
steered[:, -1, :] = steered[:, -1, :] + scale * feature_dir
```

**Example — prompt: "The new hospital opened its doors and"**

Without steering:
> The new hospital opened its doors and grew rapidly. In 1884, the small hospital was moved to the Texan capitol...

With steering (scale=12.0):
> The new hospital opened its doors and required massive amounts of money from the government of Romania. The hospital sold its doors to the people of Romania when the Communist regime was **defeated**... and they were **forced** to work. The city of Bucharest had its parks pulled down and offices and restaurants were **bombed**. The new hospital was not only a **threat** to the public health but its own people.

**Example — prompt: "The ancient city was once a thriving center of"**

Without steering:
> The ancient city was once a thriving center of trade, industry, and culture...

With steering (scale=12.0):
> The ancient city was once a thriving center of many important banking, commerce, and trade... the city of Laulaca, not only **lost** its part of its ancient city, but was also **lost** to the Roman Empire. The city was completely **destroyed** in the 3rd century BCE...

The feature coherently injects destruction/loss semantics into the generation while maintaining grammatical fluency.

**Example — suppressing the feature: "The once beautiful city was"**

Without steering:
> The once beautiful city was a swooning city, a filthy swamp but no picnic... New York was once again the scene of a massacre. There were scores of men and women slaughtered at the hands of the fire... The mass murder of New York City was the pastiche which...

With suppression (scale=-5.0):
> The once beautiful city was now grounded on solid rock, with a concrete dome over the streets... Eventually, the residents began to take more responsibility and clean up the area.

With stronger suppression (scale=-10.0):
> The once beautiful city was once home to the largest number of slaves... But in 1818, the County Council passed a bill allowing African-Americans to vote and to run the county's public schools.

Suppression doesn't eliminate negative topics entirely — the prompt still primes the model toward decline — but it consistently steers the narrative toward recovery and progress rather than dwelling on destruction. This bidirectional control (amplification and suppression) demonstrates that the SAE has learned a monosemantic, steerable representation of this concept.

## References

- Bricken et al., "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning" (Anthropic, 2023)
- Templeton et al., "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet" (Anthropic, 2024)
