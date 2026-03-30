"""
Find SAE features that activate on specific target words/concepts.

Usage:
    python find_feature.py --sae sae_checkpoint.pt --words "unreliable,doubtful,not trustworthy"
"""

import argparse
import torch
from torch.utils.data import DataLoader
from collections import defaultdict

import config
from data_utils import HFDocStream, BlockStream, ttos, stot, _train_files_for_epoch
from run_pretrained import Load_pretrained
from sae import SparseAutoencoder


CRAFTED_PROMPTS = [
    "The source was unreliable and could not be trusted.",
    "His claims were unreliable at best.",
    "The evidence was doubtful and unconvincing.",
    "She felt doubtful about the whole situation.",
    "The witness was not trustworthy at all.",
    "Many considered him untrustworthy and dishonest.",
    "The data seemed unreliable and inconsistent.",
    "There was reason to doubt his sincerity.",
    "The results were doubtful and hard to believe.",
    "Critics called the report unreliable and misleading.",
    "The promise seemed uncertain and doubtful.",
    "No one found his explanation trustworthy.",
    "The study was dismissed as unreliable by experts.",
    "Her testimony was doubtful from the start.",
    "The information came from an untrustworthy source.",
    "Skeptics questioned whether the findings were reliable.",
    "The company had a reputation for being untrustworthy.",
    "Doubts emerged about the reliability of the method.",
]

TARGET_WORDS = [
    "unreliable", "doubtful", "trustworthy", "untrustworthy",
    "doubt", "doubts", "uncertain", "skeptic", "skeptics",
    "dishonest", "misleading", "unconvincing",
]


def setup_model(args, device):
    """Load SAE and GPT-2, attach hook."""
    ckpt = torch.load(args.sae, map_location=device, weights_only=False)
    sae_cfg = ckpt["cfg"]
    hook_layer = ckpt["hook_layer"]
    hook_site = ckpt["hook_site"]
    sae = SparseAutoencoder(sae_cfg).to(device)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    print(f"SAE: {sae_cfg.d_model} -> {sae_cfg.d_sae} (layer {hook_layer}, site '{hook_site}')")

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

    return model, sae, sae_cfg, activations, handle


def is_target_token(tok_str, target_words):
    """Match only if the token IS a target word (possibly with leading space from BPE)."""
    tok_clean = tok_str.strip().lower()
    return tok_clean in target_words


def record_features(h, pos, tok_str, x, T, feature_scores):
    feat_acts = h[pos]
    active = feat_acts.nonzero(as_tuple=True)[0]
    seq_idx = pos // T
    tok_idx = pos % T
    start = max(0, tok_idx - 5)
    end = min(T, tok_idx + 6)
    context = ttos(x[seq_idx, start:end])
    for feat_idx in active:
        feat_idx = feat_idx.item()
        val = feat_acts[feat_idx].item()
        feature_scores[feat_idx].append((val, tok_str.strip(), context))


def probe_crafted_prompts(model, sae, activations, device, target_words):
    """Run crafted prompts through GPT-2 + SAE and collect features on target tokens."""
    feature_scores = defaultdict(list)
    matches = 0

    print(f"\n--- Phase 1: Probing {len(CRAFTED_PROMPTS)} crafted prompts ---")
    with torch.no_grad():
        for prompt in CRAFTED_PROMPTS:
            tokens = stot(prompt).unsqueeze(0).to(device)  # (1, T_prompt)
            model(tokens)
            acts = activations["value"]  # (1, T_prompt, d_model)
            B, T, D = acts.shape
            acts_flat = acts.reshape(-1, D)
            norms = acts_flat.norm(dim=-1, keepdim=True)
            acts_normed = acts_flat / (norms + 1e-8)
            h = sae.encode(acts_normed)
            tokens_flat = tokens.reshape(-1)

            for pos in range(tokens_flat.shape[0]):
                tok_str = ttos(torch.tensor([tokens_flat[pos].item()])).lower()
                if is_target_token(tok_str, target_words):
                    matches += 1
                    record_features(h, pos, tok_str, tokens, T, feature_scores)

    print(f"  Found {matches} target token matches across crafted prompts")
    return feature_scores, matches


def scan_corpus(model, sae, activations, device, target_words, args):
    """Scan training corpus for naturally occurring target words."""
    feature_scores = defaultdict(list)
    matches = 0

    print(f"\n--- Phase 2: Scanning {args.n_batches} batches of training data ---")
    doc_stream = HFDocStream("train", rank=0, world_size=1, limit=500, data_files=_train_files_for_epoch)
    block_stream = BlockStream(doc_stream)
    loader = DataLoader(block_stream, batch_size=args.batch_size, num_workers=0)

    with torch.no_grad():
        for i, (x, _y) in enumerate(loader):
            if i >= args.n_batches:
                break
            if i % 50 == 0:
                print(f"  Batch {i}/{args.n_batches} ({matches} matches so far)")

            x = x.to(device)
            model(x)
            acts = activations["value"]
            B, T, D = acts.shape
            acts_flat = acts.reshape(-1, D)
            norms = acts_flat.norm(dim=-1, keepdim=True)
            acts_normed = acts_flat / (norms + 1e-8)
            h = sae.encode(acts_normed)
            tokens_flat = x.reshape(-1)

            for pos in range(B * T):
                tok_str = ttos(torch.tensor([tokens_flat[pos].item()])).lower()
                if is_target_token(tok_str, target_words):
                    matches += 1
                    record_features(h, pos, tok_str, x, T, feature_scores)

    print(f"  Found {matches} target token matches in corpus")
    return feature_scores, matches


def main():
    parser = argparse.ArgumentParser(description="Find SAE features for target words")
    parser.add_argument("--sae", required=True, help="Path to SAE checkpoint")
    parser.add_argument("--checkpoint", default="myGPT2PB.pt", help="GPT-2 checkpoint path")
    parser.add_argument("--words", default=None, help="Extra comma-separated target words (added to built-in list)")
    parser.add_argument("--n_batches", type=int, default=200, help="Batches to scan from corpus")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--no_corpus", action="store_true", help="Skip corpus scan, only use crafted prompts")
    args = parser.parse_args()

    device = config.device
    target_words = list(TARGET_WORDS)
    if args.words:
        target_words += [w.strip().lower() for w in args.words.split(",")]
    print(f"Device: {device}")
    print(f"Target words: {target_words}")

    model, sae, sae_cfg, activations, handle = setup_model(args, device)

    # Phase 1: Crafted prompts (guaranteed matches)
    feature_scores, total_matches = probe_crafted_prompts(
        model, sae, activations, device, target_words)

    # Phase 2: Corpus scan (natural occurrences)
    if not args.no_corpus:
        corpus_scores, corpus_matches = scan_corpus(
            model, sae, activations, device, target_words, args)
        total_matches += corpus_matches
        for feat_idx, examples in corpus_scores.items():
            feature_scores[feat_idx].extend(examples)

    handle.remove()

    if not feature_scores:
        print("\nNo matches found for target words in the data scanned.")
        print("These words may be rare in the training data. Try increasing --n_batches.")
        return

    # Rank features by: how often they fire on target words * avg activation
    print(f"\nFound {total_matches} token matches across {len(feature_scores)} features")
    print(f"\n{'='*70}")
    print(f"Top features for: {target_words}")
    print(f"{'='*70}")

    ranked = []
    for feat_idx, examples in feature_scores.items():
        count = len(examples)
        avg_act = sum(v for v, _, _ in examples) / count
        max_act = max(v for v, _, _ in examples)
        ranked.append((feat_idx, count, avg_act, max_act, examples))

    # Sort by count * avg_activation (features that consistently fire strongly)
    ranked.sort(key=lambda r: r[1] * r[2], reverse=True)

    for feat_idx, count, avg_act, max_act, examples in ranked[:20]:
        print(f"\nFeature {feat_idx}: fired {count}x, avg_act={avg_act:.3f}, max_act={max_act:.3f}")
        for val, tok, ctx in sorted(examples, key=lambda e: e[0], reverse=True)[:3]:
            print(f"  act={val:.3f}  token={repr(tok)}  context: ...{repr(ctx)}...")


if __name__ == "__main__":
    main()
