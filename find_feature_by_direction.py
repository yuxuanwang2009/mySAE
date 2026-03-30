"""
Find SAE features by decoder alignment with a target concept direction.

Instead of matching tokens, this computes the average activation direction
for target-word contexts and finds SAE features whose decoder vectors
point in that direction.

Usage:
    python find_feature_by_direction.py --sae sae_checkpoint.pt
"""

import argparse
import torch
import torch.nn.functional as F

import config
from data_utils import stot, ttos
from run_pretrained import Load_pretrained
from sae import SparseAutoencoder


PROMPTS = [
    # Target concept: general negative tone
    ("The situation was terrible and getting worse.", "terrible"),
    ("Everything about the project was a disaster.", "disaster"),
    ("The results were disappointing and harmful.", "disappointing"),
    ("The economy suffered a devastating collapse.", "devastating"),
    ("The war caused enormous destruction and suffering.", "destruction"),
    ("The disease spread rapidly and killed thousands.", "killed"),
    ("He felt angry and bitter about the betrayal.", "bitter"),
    ("The pollution was toxic and dangerous to everyone.", "dangerous"),
    ("Crime rates surged and violence became rampant.", "violence"),
    ("The government was corrupt and incompetent.", "corrupt"),
    ("Poverty and hunger plagued the entire region.", "hunger"),
    ("The accident was horrific and left many injured.", "horrific"),
    ("Critics condemned the decision as reckless and foolish.", "reckless"),
    ("The failure was embarrassing and costly.", "embarrassing"),
    ("People were frightened and desperate for help.", "desperate"),
    ("The damage was severe and irreversible.", "severe"),
    ("The scandal destroyed his reputation completely.", "destroyed"),
    ("Conditions deteriorated and hope faded away.", "deteriorated"),
]

# Contrast prompts (positive tone)
CONTRAST_PROMPTS = [
    ("The situation was wonderful and getting better.", "wonderful"),
    ("Everything about the project was a success.", "success"),
    ("The results were impressive and beneficial.", "impressive"),
    ("The economy enjoyed a remarkable recovery.", "remarkable"),
    ("The peace brought enormous prosperity and joy.", "prosperity"),
    ("The medicine spread rapidly and saved thousands.", "saved"),
    ("He felt grateful and happy about the support.", "happy"),
    ("The environment was clean and safe for everyone.", "safe"),
    ("Crime rates dropped and harmony became widespread.", "harmony"),
    ("The government was honest and competent.", "competent"),
    ("Wealth and abundance blessed the entire region.", "abundance"),
    ("The celebration was beautiful and left many inspired.", "beautiful"),
    ("Critics praised the decision as bold and wise.", "wise"),
    ("The achievement was remarkable and rewarding.", "rewarding"),
    ("People were hopeful and confident about the future.", "confident"),
    ("The progress was steady and lasting.", "steady"),
    ("The triumph elevated his reputation completely.", "elevated"),
    ("Conditions improved and hope flourished.", "improved"),
]


def find_token_position(tokens_1d, target_word):
    """Find the position of the target word in the token sequence."""
    # Decode each token and find the one matching target_word
    for i in range(tokens_1d.shape[0]):
        tok_str = ttos(tokens_1d[i:i+1]).strip()
        if tok_str.lower() == target_word.lower():
            return i
    # Fallback: partial match
    for i in range(tokens_1d.shape[0]):
        tok_str = ttos(tokens_1d[i:i+1]).strip().lower()
        if target_word.lower() in tok_str or tok_str in target_word.lower():
            if len(tok_str) >= 4:
                return i
    return None


def collect_directions(model, activations, prompts, device):
    """Collect activation vectors at target word positions."""
    directions = []
    with torch.no_grad():
        for prompt, target_word in prompts:
            tokens = stot(prompt).unsqueeze(0).to(device)
            model(tokens)
            acts = activations["value"]  # (1, T, d_model)

            pos = find_token_position(tokens[0], target_word)
            if pos is not None:
                act_vec = acts[0, pos]  # (d_model,)
                directions.append(act_vec)
            else:
                print(f"  Warning: couldn't find '{target_word}' in '{prompt}'")
    return directions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae", required=True, help="Path to SAE checkpoint")
    parser.add_argument("--checkpoint", default="myGPT2PB.pt")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top features to show")
    parser.add_argument("--no_contrast", action="store_true", help="Skip contrastive subtraction")
    args = parser.parse_args()

    device = config.device
    print(f"Device: {device}")

    # Load SAE
    ckpt = torch.load(args.sae, map_location=device, weights_only=False)
    sae_cfg = ckpt["cfg"]
    hook_layer = ckpt["hook_layer"]
    hook_site = ckpt["hook_site"]
    sae = SparseAutoencoder(sae_cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    print(f"SAE: {sae_cfg.d_model} -> {sae_cfg.d_sae} (layer {hook_layer}, site '{hook_site}')")

    # Load GPT-2
    model = Load_pretrained(args.checkpoint, device=device)
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

    # Collect target directions
    print("\nCollecting target concept activations...")
    target_dirs = collect_directions(model, activations, PROMPTS, device)
    print(f"  Collected {len(target_dirs)} target vectors")
    target_mean = torch.stack(target_dirs).mean(dim=0)  # (d_model,)

    if not args.no_contrast:
        print("Collecting contrast activations...")
        contrast_dirs = collect_directions(model, activations, CONTRAST_PROMPTS, device)
        print(f"  Collected {len(contrast_dirs)} contrast vectors")
        contrast_mean = torch.stack(contrast_dirs).mean(dim=0)

        # The concept direction = target - contrast
        concept_dir = target_mean - contrast_mean
        print("Using contrastive direction (target - contrast)")
    else:
        concept_dir = target_mean
        print("Using raw target direction (no contrast)")

    concept_dir = F.normalize(concept_dir, dim=0)  # unit norm

    handle.remove()

    # Compare with SAE decoder columns
    # W_dec is (d_model, d_sae) — each column is a feature's decoder direction
    W_dec = sae.W_dec.data  # (d_model, d_sae)
    W_dec_normed = F.normalize(W_dec, dim=0)  # normalize each column

    # Cosine similarity between concept direction and each decoder column
    similarities = concept_dir @ W_dec_normed  # (d_sae,)

    # Top features (most aligned)
    top_vals, top_idx = similarities.topk(args.top_k)

    print(f"\n{'='*60}")
    print(f"Top {args.top_k} features aligned with NEGATIVE tone direction")
    print(f"{'='*60}")
    for i in range(args.top_k):
        feat = top_idx[i].item()
        sim = top_vals[i].item()
        print(f"  Feature {feat:5d}  cosine={sim:.4f}")

    # Also show bottom (most anti-aligned = "reliable/trustworthy" features)
    bot_vals, bot_idx = similarities.topk(args.top_k, largest=False)
    print(f"\n{'='*60}")
    print(f"Top {args.top_k} features aligned with POSITIVE tone direction")
    print(f"{'='*60}")
    for i in range(args.top_k):
        feat = bot_idx[i].item()
        sim = bot_vals[i].item()
        print(f"  Feature {feat:5d}  cosine={sim:.4f}")


if __name__ == "__main__":
    main()
