"""
Automated interpretability for SAE features using Claude.

Collects top activating examples for a feature, asks Claude to describe
the pattern, then validates the description on held-out examples.

Usage:
    python interpret.py --sae sae_checkpoint.pt --feature 42
    python interpret.py --sae sae_checkpoint.pt --feature 42 --n_batches 100
"""

import argparse
import random
import torch
from torch.utils.data import DataLoader

import anthropic

import config
from data_utils import HFDocStream, BlockStream, ttos, _train_files_for_epoch
from run_pretrained import Load_pretrained
from sae import SparseAutoencoder


def load_sae(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sae_cfg = ckpt["cfg"]
    sae = SparseAutoencoder(sae_cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt["hook_layer"], ckpt["hook_site"]


def setup_model_and_hook(checkpoint, hook_layer, hook_site, device):
    model = Load_pretrained(checkpoint, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

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
    doc_stream = HFDocStream("train", rank=0, world_size=1, limit=200, data_files=_train_files_for_epoch)
    block_stream = BlockStream(doc_stream)
    return DataLoader(block_stream, batch_size=batch_size, num_workers=0)


def collect_examples(model, sae, activations, device, feature_idx, n_batches=50):
    """Collect all activating examples for a feature."""
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
    return results


def collect_non_examples(model, sae, activations, device, feature_idx, n=10):
    """Collect examples where the feature does NOT fire."""
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
            if len(non_examples) >= n:
                break

    return non_examples[:n]


def interpret(feature_idx, top_examples, random_examples, non_examples):
    """Ask Claude to interpret and validate the feature."""
    client = anthropic.Anthropic()

    # Step 1: Describe
    examples_text = "\n".join(
        f"  act={val:.2f}  token={repr(tok)}  context: ...{repr(ctx)}..."
        for val, tok, ctx in top_examples
    )

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

    # Step 2: Validate
    if not random_examples:
        return description, None

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

    return description, accuracy


def main():
    parser = argparse.ArgumentParser(description="Automated SAE feature interpretability")
    parser.add_argument("--sae", required=True, help="Path to SAE checkpoint")
    parser.add_argument("--checkpoint", default="myGPT2PB.pt", help="GPT-2 checkpoint path")
    parser.add_argument("--feature", type=int, required=True, help="Feature index to interpret")
    parser.add_argument("--n_batches", type=int, default=50, help="Batches to collect examples from")
    args = parser.parse_args()

    device = config.device
    print(f"Device: {device}")

    sae, hook_layer, hook_site = load_sae(args.sae, device)
    print(f"SAE: {sae.cfg.d_model} -> {sae.cfg.d_sae} (layer {hook_layer}, site '{hook_site}')")

    model, activations, handle = setup_model_and_hook(args.checkpoint, hook_layer, hook_site, device)

    try:
        results = collect_examples(model, sae, activations, device, args.feature, args.n_batches)
        top_examples = results[:20]
        random_examples = results[len(results)//3 : len(results)//3 + 10] if len(results) > 30 else []
        non_examples = collect_non_examples(model, sae, activations, device, args.feature)

        interpret(args.feature, top_examples, random_examples, non_examples)
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
