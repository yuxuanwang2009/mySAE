# run_pretrained.py
import argparse
import torch
import config
from train_utils import Construct_optimizer
from data_utils import ttos, stot
from model import GPTLanguageModel
from pathlib import Path

# Utility functions
def Load_pretrained(checkpoint_path: str = "checkpoint.pt", training = False, device = config.device) -> GPTLanguageModel:
    if checkpoint_path == "GPT2.pt":
        config.cfg.vocab_size = 50257
    model = GPTLanguageModel(cfg=config.cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)    
    model.load_state_dict(ckpt["model"])
    if training == True:
        optimizer = Construct_optimizer(model, config.lr, config.weight_decay, device)
        optimizer.load_state_dict(ckpt["optimizer"])
        model.train()
        return model, optimizer
    else:
        model.eval()
        return model

def generate_words(prompt: str, model: GPTLanguageModel, max_new_tokens: int = 3000, beta: float = 1.0):
    with torch.no_grad():
        generated = model.generate_with_cache(
            idx=prompt,
            max_new_tokens=max_new_tokens,
            beta=beta,
        )
    words_gen_string = ttos(generated.view(-1), for_output=True)
    return words_gen_string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='myGPT2PB.pt', help='Enter your pretrained model checkpoint')
    args = parser.parse_args()
    checkpoint_path = args.model
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    elif checkpoint_path == "GPT2.pt":
        config.cfg.vocab = 50257
    model = Load_pretrained(checkpoint_path)
    # model = GPTLanguageModel(cfg=config.cfg).to(config.device)
    # model = GPTLanguageModel.load_gpt2_from_hf().to(config.device) # for testing imported GPT-2, temporary

    
    while True:
        user_prompt = input("Enter your prompt (or '<q>' to quit): ")
        if user_prompt == "<q>":
            break
        if user_prompt == "":
            user_prompt = "<|endoftext|>"

        prompt = stot(user_prompt).view(1, -1)
        prompt = prompt.to(config.device)
        # torch.manual_seed(42)
        # torch.mps.manual_seed(42)
        words_gen_string = generate_words(prompt, model, max_new_tokens=500, beta=1)

        print(f"\n{words_gen_string}")
        with open("generated.txt", "w") as f:
            f.write(words_gen_string)
        print("\nSaved to generated.txt.")

if __name__ == "__main__":
        main()
        
