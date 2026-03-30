"""
Evaluate and analyze a trained Sparse Autoencoder.

Usage:
    python eval_sae.py --sae sae_checkpoint.pt --mode stats
    python eval_sae.py --sae sae_checkpoint.pt --mode top_tokens --feature 42
    python eval_sae.py --sae sae_checkpoint.pt --mode density

See sae_tutorial.md for how to interpret the output.
"""

import argparse
import torch
from torch.utils.data import DataLoader

import config
from data_utils import HFDocStream, BlockStream, ttos, stot, _train_files_for_epoch
from run_pretrained import Load_pretrained
from sae import SparseAutoencoder, SAEConfig


def load_sae(path, device):
    """Load a trained SAE from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sae_cfg = ckpt["cfg"]
    sae = SparseAutoencoder(sae_cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt["hook_layer"], ckpt["hook_site"]


def setup_model_and_hook(checkpoint, hook_layer, hook_site, device):
    """Load frozen GPT-2 and attach activation hook."""
    model = Load_pretrained(checkpoint, device=device)
    model.eval()

    activations = {}
    def hook_fn(module, input, output):
        activations["value"] = output.detach()

    block = model.blocks[hook_layer]
    if hook_site == "residual":
        target = block
    elif hook_site == "mlp":
        target = block.ffd[1]
    elif hook_site == "attn":
        target = block.atten[1]

    handle = target.register_forward_hook(hook_fn)
    return model, activations, handle


def get_data_loader(batch_size=4):
    """Create a small data loader for evaluation."""
    doc_stream = HFDocStream("train", rank=0, world_size=1, limit=200, data_files=_train_files_for_epoch)
    block_stream = BlockStream(doc_stream)
    return DataLoader(block_stream, batch_size=batch_size, num_workers=0)


# ── Mode: reconstruction stats ──────────────────────────────────────────

def reconstruction_stats(model, sae, activations, device, n_batches=50):
    """Compute aggregate reconstruction metrics."""
    loader = get_data_loader()

    total_mse = 0.0
    total_l0 = 0.0
    total_var_explained = 0.0
    total_dead = torch.zeros(sae.cfg.d_sae, device=device)  # track which features ever fire
    count = 0

    with torch.no_grad():
        for i, (x, _y) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device)
            model(x)
            acts = activations["value"].reshape(-1, sae.cfg.d_model)
            acts_norm = acts.norm(dim=-1, keepdim=True)
            acts_normed = acts / (acts_norm + 1e-8)

            _, mse, _, h, x_hat = sae(acts_normed)
            x_hat_rescaled = x_hat * acts_norm

            total_mse += mse.item()
            total_l0 += (h > 0).float().mean().item() * sae.cfg.d_sae
            total_var_explained += (1 - (acts - x_hat_rescaled).var() / acts.var()).item()
            total_dead += (h > 0).any(dim=0).float()
            count += 1

    n = max(count, 1)
    alive = (total_dead > 0).sum().item()
    dead = sae.cfg.d_sae - alive

    print(f"\n{'='*50}")
    print(f"Reconstruction Stats ({count} batches)")
    print(f"{'='*50}")
    print(f"  MSE:              {total_mse / n:.4f}")
    print(f"  Explained var:    {total_var_explained / n:.4f}")
    print(f"  Avg L0:           {total_l0 / n:.1f} / {sae.cfg.d_sae}")
    print(f"  Alive features:   {alive} / {sae.cfg.d_sae} ({100 * alive / sae.cfg.d_sae:.1f}%)")
    print(f"  Dead features:    {dead} / {sae.cfg.d_sae} ({100 * dead / sae.cfg.d_sae:.1f}%)")


# ── Mode: top activating tokens ─────────────────────────────────────────

def top_activating_tokens(model, sae, activations, device, feature_idx, top_k=20, n_batches=50):
    """Find the tokens that most strongly activate a given SAE feature."""
    loader = get_data_loader()

    # Collect (activation_value, token_id, context_ids) tuples
    results = []

    with torch.no_grad():
        for i, (x, _y) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device)  # (B, T)
            model(x)
            acts = activations["value"].reshape(-1, sae.cfg.d_model)  # (B*T, d_model)
            h = sae.encode(acts)  # (B*T, d_sae)

            feature_acts = h[:, feature_idx]  # (B*T,)
            tokens_flat = x.reshape(-1)       # (B*T,)

            # Find positions where this feature fires
            nonzero_mask = feature_acts > 0
            if nonzero_mask.any():
                values = feature_acts[nonzero_mask]
                positions = nonzero_mask.nonzero(as_tuple=True)[0]

                for val, pos in zip(values, positions):
                    pos_int = pos.item()
                    # Grab surrounding context (5 tokens before and after)
                    B, T = x.shape[0], x.shape[1]
                    seq_idx = pos_int // T
                    tok_idx = pos_int % T
                    start = max(0, tok_idx - 5)
                    end = min(T, tok_idx + 6)
                    context = x[seq_idx, start:end]
                    results.append((val.item(), tokens_flat[pos_int].item(), tok_idx, context))

    # Sort by activation value, take top-k
    results.sort(key=lambda r: r[0], reverse=True)
    results = results[:top_k]

    print(f"\n{'='*60}")
    print(f"Feature {feature_idx} — Top {len(results)} activating tokens")
    print(f"{'='*60}")
    for val, token_id, tok_pos, context in results:
        token_str = ttos(torch.tensor([token_id]))
        context_str = ttos(context)
        # Mark the target token in context
        print(f"  act={val:>7.2f}  token={repr(token_str):<15s}  context: ...{repr(context_str)}...")


# ── Mode: feature density ───────────────────────────────────────────────

def feature_density(model, sae, activations, device, n_batches=50):
    """Compute how often each feature activates."""
    loader = get_data_loader()

    fire_counts = torch.zeros(sae.cfg.d_sae, device=device)
    total_tokens = 0

    with torch.no_grad():
        for i, (x, _y) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device)
            model(x)
            acts = activations["value"].reshape(-1, sae.cfg.d_model)
            h = sae.encode(acts)

            fire_counts += (h > 0).float().sum(dim=0)
            total_tokens += acts.shape[0]

    freqs = fire_counts / max(total_tokens, 1)

    # Print histogram of firing frequencies
    print(f"\n{'='*50}")
    print(f"Feature Density ({total_tokens:,} tokens)")
    print(f"{'='*50}")

    bins = [0, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0]
    for lo, hi in zip(bins[:-1], bins[1:]):
        count = ((freqs > lo) & (freqs <= hi)).sum().item()
        print(f"  ({lo:.0e}, {hi:.0e}]: {count:>6d} features")
    dead = (freqs == 0).sum().item()
    print(f"  dead (never fired): {dead:>6d} features")

    # Top 10 most active features
    top_vals, top_idxs = freqs.topk(10)
    print(f"\nTop 10 most active features:")
    for idx, freq in zip(top_idxs, top_vals):
        print(f"  Feature {idx.item():<6d}  fires {100 * freq.item():.2f}% of tokens")


# ── Mode: interpret ────────────────────────────────────────────────────

def interpret_feature(model, sae, activations, device, feature_idx, n_batches=50):
    """Use Claude to automatically interpret what a feature represents."""
    import anthropic

    # Step 1: Collect top activating examples
    loader = get_data_loader()
    results = []

    with torch.no_grad():
        for i, (x, _y) in enumerate(loader):
            if i >= n_batches:
                break
            x = x.to(device)
            model(x)
            acts = activations["value"].reshape(-1, sae.cfg.d_model)
            h = sae.encode(acts)

            feature_acts = h[:, feature_idx]
            tokens_flat = x.reshape(-1)

            nonzero_mask = feature_acts > 0
            if nonzero_mask.any():
                values = feature_acts[nonzero_mask]
                positions = nonzero_mask.nonzero(as_tuple=True)[0]

                for val, pos in zip(values, positions):
                    pos_int = pos.item()
                    B, T = x.shape[0], x.shape[1]
                    seq_idx = pos_int // T
                    tok_idx = pos_int % T
                    start = max(0, tok_idx - 5)
                    end = min(T, tok_idx + 6)
                    context = x[seq_idx, start:end]
                    token_str = ttos(torch.tensor([tokens_flat[pos_int].item()]))
                    context_str = ttos(context)
                    results.append((val.item(), token_str, context_str))

    results.sort(key=lambda r: r[0], reverse=True)
    top_examples = results[:20]
    random_examples = results[len(results)//3 : len(results)//3 + 10] if len(results) > 30 else []

    # Step 2: Ask Claude to describe the feature
    examples_text = "\n".join(
        f"  act={val:.2f}  token={repr(tok)}  context: ...{repr(ctx)}..."
        for val, tok, ctx in top_examples
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                "You are analyzing a feature from a sparse autoencoder trained on GPT-2 activations. "
                "Below are the top activating examples for this feature. Each shows the activation strength, "
                "the target token, and surrounding context.\n\n"
                f"Feature {feature_idx} — Top activating examples:\n{examples_text}\n\n"
                "In 1-2 sentences, describe what concept or pattern this feature captures. "
                "Be specific and concise."
            )
        }]
    )
    description = response.content[0].text

    print(f"\n{'='*60}")
    print(f"Feature {feature_idx} — Automated Interpretation")
    print(f"{'='*60}")
    print(f"\nTop 5 examples:")
    for val, tok, ctx in top_examples[:5]:
        print(f"  act={val:.2f}  token={repr(tok)}  context: ...{repr(ctx)}...")
    print(f"\nInterpretation: {description}")

    # Step 3: Validate — ask Claude to predict on held-out examples
    if random_examples:
        held_out_text = "\n".join(
            f"  {i+1}. token={repr(tok)}  context: ...{repr(ctx)}..."
            for i, (_, tok, ctx) in enumerate(random_examples)
        )

        # Also collect some examples where the feature did NOT fire
        non_examples = []
        with torch.no_grad():
            for i, (x, _y) in enumerate(get_data_loader()):
                if i >= 5:
                    break
                x = x.to(device)
                model(x)
                acts = activations["value"].reshape(-1, sae.cfg.d_model)
                h = sae.encode(acts)
                feature_acts = h[:, feature_idx]
                tokens_flat = x.reshape(-1)

                zero_mask = feature_acts == 0
                if zero_mask.any():
                    positions = zero_mask.nonzero(as_tuple=True)[0][:5]
                    for pos in positions:
                        pos_int = pos.item()
                        B, T = x.shape[0], x.shape[1]
                        seq_idx = pos_int // T
                        tok_idx = pos_int % T
                        start = max(0, tok_idx - 5)
                        end = min(T, tok_idx + 6)
                        context = x[seq_idx, start:end]
                        tok = ttos(torch.tensor([tokens_flat[pos_int].item()]))
                        ctx = ttos(context)
                        non_examples.append((tok, ctx))
                if len(non_examples) >= 10:
                    break

        # Mix positive and negative examples
        import random
        test_items = [(tok, ctx, True) for _, tok, ctx in random_examples]
        test_items += [(tok, ctx, False) for tok, ctx in non_examples[:10]]
        random.shuffle(test_items)

        test_text = "\n".join(
            f"  {i+1}. token={repr(tok)}  context: ...{repr(ctx)}..."
            for i, (tok, ctx, _) in enumerate(test_items)
        )

        val_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": (
                    f"A sparse autoencoder feature has been described as:\n\"{description}\"\n\n"
                    "For each example below, predict YES if the feature would activate, NO if not. "
                    "Reply with just the number and YES/NO for each.\n\n"
                    f"{test_text}"
                )
            }]
        )
        predictions = val_response.content[0].text

        # Score predictions
        lines = [l.strip() for l in predictions.strip().split("\n") if l.strip()]
        correct = 0
        total = 0
        for line, (_, _, label) in zip(lines, test_items):
            predicted_yes = "YES" in line.upper()
            if predicted_yes == label:
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        print(f"\nValidation: {correct}/{total} correct ({accuracy:.0%})")
        print(f"Predictions:\n{predictions}")


# ── Mode: token features ───────────────────────────────────────────────

def token_features(model, sae, activations, device, text, token_pos, top_k=20):
    """Show which SAE features activate most for a specific token position in a sentence."""
    tokens = stot(text)
    x = tokens.unsqueeze(0).to(device)  # (1, T)

    with torch.no_grad():
        model(x)
        acts = activations["value"]  # (1, T, d_model)
        act_vec = acts[0, token_pos].unsqueeze(0)  # (1, d_model)
        act_norm = act_vec.norm(dim=-1, keepdim=True)
        act_normed = act_vec / (act_norm + 1e-8)

        h = sae.encode(act_normed)  # (1, d_sae)

    # Show the sentence with the target token highlighted
    token_list = [ttos(tokens[i:i+1]) for i in range(len(tokens))]
    print(f"\nText: {text}")
    print(f"Tokens: {token_list}")
    print(f"Target: position {token_pos} = {repr(token_list[token_pos])}")

    # Top-k active features
    vals, idxs = h[0].topk(top_k)
    print(f"\n{'='*50}")
    print(f"Top {top_k} features at position {token_pos}")
    print(f"{'='*50}")
    for idx, val in zip(idxs, vals):
        if val.item() <= 0:
            break
        print(f"  Feature {idx.item():<6d}  activation={val.item():.4f}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAE")
    parser.add_argument("--sae", required=True, help="Path to SAE checkpoint")
    parser.add_argument("--checkpoint", default="myGPT2PB.pt", help="GPT-2 checkpoint path")
    parser.add_argument("--mode", required=True, choices=["stats", "top_tokens", "density", "token_features", "interpret"],
                        help="What to evaluate")
    parser.add_argument("--feature", type=int, default=0, help="Feature index for top_tokens mode")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top results to show")
    parser.add_argument("--text", type=str, default=None, help="Input text for token_features mode")
    parser.add_argument("--token_pos", type=int, default=-1, help="Token position to inspect (default: last)")
    parser.add_argument("--n_batches", type=int, default=50, help="Number of data batches to evaluate on")
    args = parser.parse_args()

    device = config.device
    print(f"Device: {device}")

    # Load SAE and GPT-2
    sae, hook_layer, hook_site = load_sae(args.sae, device)
    print(f"SAE loaded: {sae.cfg.d_model} -> {sae.cfg.d_sae} (layer {hook_layer}, site '{hook_site}')")

    model, activations, handle = setup_model_and_hook(args.checkpoint, hook_layer, hook_site, device)

    try:
        if args.mode == "stats":
            reconstruction_stats(model, sae, activations, device, n_batches=args.n_batches)
        elif args.mode == "top_tokens":
            top_activating_tokens(model, sae, activations, device,
                                  feature_idx=args.feature, top_k=args.top_k, n_batches=args.n_batches)
        elif args.mode == "density":
            feature_density(model, sae, activations, device, n_batches=args.n_batches)
        elif args.mode == "token_features":
            token_features(model, sae, activations, device,
                           text=args.text, token_pos=args.token_pos, top_k=args.top_k)
        elif args.mode == "interpret":
            interpret_feature(model, sae, activations, device,
                              feature_idx=args.feature, n_batches=args.n_batches)
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
