"""
Train a Sparse Autoencoder on GPT-2 activations.

Usage:
    python train_sae.py                          # defaults: layer 6, 6144 features, 10M tokens
    python train_sae.py --hook_layer 3 --l1_coeff 1e-3 --total_tokens 50000000

See sae_tutorial.md for a conceptual walkthrough.
"""

import argparse
import time
import torch
from torch.utils.data import DataLoader

import config
from data_utils import HFDocStream, BlockStream, _train_files_for_epoch
from run_pretrained import Load_pretrained
from sae import SparseAutoencoder, SAEConfig


def main():
    parser = argparse.ArgumentParser(description="Train SAE on GPT-2 activations")
    parser.add_argument("--checkpoint", default="myGPT2PB.pt", help="GPT-2 checkpoint path")
    parser.add_argument("--hook_layer", type=int, default=6, help="Which transformer block to hook (0-11)")
    parser.add_argument("--hook_site", default="residual", choices=["residual", "mlp", "attn"],
                        help="Where to hook: residual stream, MLP output, or attention output")
    parser.add_argument("--d_sae", type=int, default=768 * 8, help="SAE hidden dimension")
    parser.add_argument("--l1_coeff", type=float, default=5e-3, help="L1 sparsity penalty")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gpt2_batch_size", type=int, default=8, help="Sequences per GPT-2 forward pass")
    parser.add_argument("--total_tokens", type=int, default=10_000_000, help="Total tokens to train on")
    parser.add_argument("--log_interval", type=int, default=100, help="Print stats every N SAE steps")
    parser.add_argument("--norm_interval", type=int, default=100, help="Normalize decoder every N steps")
    parser.add_argument("--save_path", default="sae_checkpoint.pt", help="Where to save the trained SAE")
    args = parser.parse_args()

    device = config.device
    print(f"Device: {device}")

    # ── 1. Load frozen GPT-2 ────────────────────────────────────────────
    print(f"Loading GPT-2 from {args.checkpoint}...")
    model = Load_pretrained(args.checkpoint, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("GPT-2 loaded and frozen.")

    # ── 2. Attach hook ──────────────────────────────────────────────────
    # This is all it takes to intercept activations — no wrapper class needed.
    # The mutatable wrapper (dict) here ensures activation gets populated by hook_fn.
    # The naive approach would only trigger local binding, which does not get carried
    # outside the function. 
    activations = {}

    def hook_fn(module, input, output):
        activations["value"] = output.detach()

    # Pick the right module to hook
    block = model.blocks[args.hook_layer]
    if args.hook_site == "residual":
        target_module = block                   # full TransformerBlock output
    elif args.hook_site == "mlp":
        target_module = block.ffd[1]            # FeedForward module (inside Sequential between LN and Dropout)
    elif args.hook_site == "attn":
        target_module = block.atten[1]          # MultiHeadAttention module 

    handle = target_module.register_forward_hook(hook_fn)
    print(f"Hook attached: layer {args.hook_layer}, site '{args.hook_site}'")

    # ── 3. Create SAE ──────────────────────────────────────────────────
    sae_cfg = SAEConfig(d_model=config.n_emb, d_sae=args.d_sae, l1_coeff=args.l1_coeff)
    sae = SparseAutoencoder(sae_cfg).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)
    print(f"SAE: {config.n_emb} -> {args.d_sae} ({args.d_sae // config.n_emb}x expansion)")
    print(f"SAE parameters: {sum(p.numel() for p in sae.parameters()) / 1e6:.1f}M")

    # ── 4. Data stream ─────────────────────────────────────────────────
    doc_stream = HFDocStream("train", rank=0, world_size=1, limit=None, data_files=_train_files_for_epoch)
    block_stream = BlockStream(doc_stream)
    loader = DataLoader(block_stream, batch_size=args.gpt2_batch_size, num_workers=0)

    # ── 5. Training loop ───────────────────────────────────────────────
    tokens_seen = 0
    sae_step = 0
    running_mse = 0.0
    running_l1 = 0.0
    running_l0 = 0.0
    t0 = time.time()

    print(f"\nTraining for {args.total_tokens:,} tokens...")
    print(f"{'step':>8} {'tokens':>12} {'mse':>10} {'l1':>10} {'L0':>8} {'expl_var':>10} {'tok/s':>10}")
    print("-" * 76)

    sae.train()
    for x, _y in loader:
        x = x.to(device)  # (B, T) token ids

        # Forward through frozen GPT-2 — the hook captures activations
        with torch.no_grad():
            model(x)

        # Grab activations: (B, T, d_model) -> (B*T, d_model)
        acts = activations["value"]
        acts = acts.reshape(-1, config.n_emb)  # each token position is independent

        # SAE forward + backward
        loss, mse, l1, h, x_hat = sae(acts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize decoder columns periodically
        if sae_step % args.norm_interval == 0:
            with torch.no_grad():
                sae.W_dec.data /= sae.W_dec.data.norm(dim=0, keepdim=True)

        # Track stats
        with torch.no_grad():
            l0 = (h > 0).float().mean().item() * sae_cfg.d_sae  # avg active features
        running_mse += mse.item()
        running_l1 += l1.item()
        running_l0 += l0
        sae_step += 1
        tokens_seen += acts.shape[0]

        # Log
        if sae_step % args.log_interval == 0:
            avg_mse = running_mse / args.log_interval
            avg_l1 = running_l1 / args.log_interval
            avg_l0 = running_l0 / args.log_interval
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / elapsed

            # Explained variance on last batch
            with torch.no_grad():
                expl_var = 1 - (acts - x_hat).var() / acts.var()

            print(f"{sae_step:>8d} {tokens_seen:>12,} {avg_mse:>10.4f} {avg_l1:>10.4f} "
                  f"{avg_l0:>8.1f} {expl_var.item():>10.4f} {tok_per_sec:>10.0f}")

            running_mse = 0.0
            running_l1 = 0.0
            running_l0 = 0.0

        # Done?
        if tokens_seen >= args.total_tokens:
            break

    # ── 6. Save ─────────────────────────────────────────────────────────
    handle.remove()  # clean up the hook

    save_dict = {
        "state_dict": sae.state_dict(),
        "cfg": sae_cfg,
        "hook_layer": args.hook_layer,
        "hook_site": args.hook_site,
        "tokens_trained": tokens_seen,
    }
    torch.save(save_dict, args.save_path)
    print(f"\nSaved SAE to {args.save_path}")
    print(f"Trained on {tokens_seen:,} tokens in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
