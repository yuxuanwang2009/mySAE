"""
Steer GPT-2 generation by amplifying or suppressing SAE features.

Usage:
    python steer.py --sae sae_checkpoint.pt --feature 42 --scale 5.0 --prompt "The doctor explained that"
    python steer.py --sae sae_checkpoint.pt --feature 42 --scale -5.0 --prompt "The doctor explained that"
"""

import argparse
import torch

import config
from data_utils import stot, ttos
from run_pretrained import Load_pretrained
from sae import SparseAutoencoder


def main():
    parser = argparse.ArgumentParser(description="Steer GPT-2 with SAE features")
    parser.add_argument("--sae", required=True, help="Path to SAE checkpoint")
    parser.add_argument("--checkpoint", default="myGPT2PB.pt", help="GPT-2 checkpoint path")
    parser.add_argument("--feature", type=int, required=True, help="Feature index to steer")
    parser.add_argument("--scale", type=float, default=5.0, help="Steering strength (negative to suppress)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    device = config.device

    # Load GPT-2
    model = Load_pretrained(args.checkpoint, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load SAE
    ckpt = torch.load(args.sae, map_location=device, weights_only=False)
    sae_cfg = ckpt["cfg"]
    hook_layer = ckpt["hook_layer"]
    hook_site = ckpt["hook_site"]
    sae = SparseAutoencoder(sae_cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()

    print(f"Steering feature {args.feature} with scale {args.scale}")
    print(f"Hook: layer {hook_layer}, site '{hook_site}'")

    # Steering hook: add the feature's decoder direction directly to the residual stream
    # This avoids SAE reconstruction error — just nudges the activation
    feature_dir = sae.W_dec[:, args.feature].detach()  # (d_model,)
    print(f"Decoder direction norm: {feature_dir.norm().item():.4f}")

    def steering_hook(module, input, output):
        # Only steer the last token position (the one being generated)
        steered = output.clone()
        residual_norm = steered[:, -1, :].norm().item()
        steered[:, -1, :] = steered[:, -1, :] + args.scale * feature_dir
        if not hasattr(steering_hook, '_printed'):
            print(f"Residual stream norm at last token: {residual_norm:.2f}")
            print(f"Steering magnitude: {args.scale:.2f} ({args.scale/residual_norm:.1%} of residual)")
            steering_hook._printed = True
        return steered

    # Attach steering hook
    block = model.blocks[hook_layer]
    if hook_site == "residual":
        target = block
    elif hook_site == "mlp":
        target = block.ffd[1]
    elif hook_site == "attn":
        target = block.atten[1]
    handle = target.register_forward_hook(steering_hook)

    # Generate with steering
    tokens = stot(args.prompt).unsqueeze(0).to(device)
    print(f"\nPrompt: {args.prompt}")
    print(f"{'='*60}")
    print(args.prompt, end="", flush=True)

    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits, _ = model(tokens)  # (1, T, vocab)
            next_logits = logits[0, -1] / args.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # Stop on end of text
            if next_token.item() == stot("<|endoftext|>")[0].item():
                break

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            print(ttos(next_token), end="", flush=True)

    handle.remove()

    # Also generate without steering for comparison
    print(f"\n\n{'='*60}")
    print("Without steering:")
    print(f"{'='*60}")
    print(args.prompt, end="", flush=True)

    tokens = stot(args.prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(args.max_tokens):
            logits, _ = model(tokens)
            next_logits = logits[0, -1] / args.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            if next_token.item() == stot("<|endoftext|>")[0].item():
                break

            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            print(ttos(next_token), end="", flush=True)

    print()


if __name__ == "__main__":
    main()
