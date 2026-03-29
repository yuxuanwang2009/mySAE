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
from data_utils import HFDocStream, BlockStream, ttos, _train_files_for_epoch
from run_pretrained import Load_pretrained
from sae import SparseAutoencoder, SAEConfig


def load_sae(path, device):
    """Load a trained SAE from checkpoint."""
    ckpt = torch.load(path, map_location=device)
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

            _, mse, _, h, x_hat = sae(acts)

            total_mse += mse.item()
            total_l0 += (h > 0).float().mean().item() * sae.cfg.d_sae
            total_var_explained += (1 - (acts - x_hat).var() / acts.var()).item()
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


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAE")
    parser.add_argument("--sae", required=True, help="Path to SAE checkpoint")
    parser.add_argument("--checkpoint", default="myGPT2PB.pt", help="GPT-2 checkpoint path")
    parser.add_argument("--mode", required=True, choices=["stats", "top_tokens", "density"],
                        help="What to evaluate")
    parser.add_argument("--feature", type=int, default=0, help="Feature index for top_tokens mode")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top tokens to show")
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
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
