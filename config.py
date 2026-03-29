from __future__ import annotations

from dataclasses import dataclass, field

import torch


def _default_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


@dataclass
class Config:
    """Training/model configuration. Can construct other presets.
    """
    # model
    n_emb: int = 768
    T: int = 1024  # context size
    vocab_size: int = 50304  # or 50304 padded up to nearest multiple of 64 for efficiency
    n_layers: int = 12
    n_heads: int = 12
    bias: bool = True  # same value in GPT-2, but False is better
    dropout: float = 0.0  # GPT-2 value
    label_smoothing: float = 0.0
    weight_tying: bool = True

    device: torch.device = field(default_factory=_default_device)
    n_ffd_hidden: int = field(init=False)

    # data
    max_steps: int = 70000  # how many batches to train for
    eval_interval: int = 600
    batch_size: int = 64
    macro_batch_size: int = 512  # for gradient accumulation to simulate larger batch sizes

    # optimizer
    lr: float = 6e-4
    min_lr: float = 0.1 * lr
    warmup_ratio: float = 0.04
    weight_decay: float = 0.1  # GPT-2 value
    grad_clipping: float = 1.0  # gradient norm clipping

    # lr scheduler
    scheduler: str = "cosine"  # "cosine" or "plateau"

    # reproducibility
    seed: int = 1337

    # tokenizer
    use_tiktoken: bool = False

    def __post_init__(self) -> None:
        self.n_ffd_hidden = 4 * self.n_emb
        assert self.macro_batch_size % self.batch_size == 0
    
    @classmethod
    def tiny(cls) -> "Config":
        return cls(
            n_emb=120,
            n_layers=6,
            n_heads=6,
            T=64,
            vocab_size=3072,
            use_tiktoken=False,
            dropout=0.2,
            batch_size=128,
            macro_batch_size=256,
            bias=False,
            lr = 6e-4,
            min_lr=2e-6,
            scheduler="cosine",
            max_steps= 18000, # in terms of macrobatches
            eval_interval=600, # in terms of macrobatches
            warmup_ratio=0.04,
            weight_decay=0.02,
            weight_tying=False,
            grad_clipping=3.0
        )

cfg = Config()

# Backwards-compatible module-level exports, DO NOT DELETE
n_emb = cfg.n_emb
T = cfg.T
vocab_size = cfg.vocab_size
n_layers = cfg.n_layers
n_heads = cfg.n_heads
n_ffd_hidden = cfg.n_ffd_hidden
bias = cfg.bias
dropout = cfg.dropout
label_smoothing = cfg.label_smoothing
weight_tying = cfg.weight_tying
device = cfg.device

max_steps = cfg.max_steps
eval_interval = cfg.eval_interval
batch_size = cfg.batch_size
macro_batch_size = cfg.macro_batch_size

lr = cfg.lr
min_lr = cfg.min_lr
warmup_ratio = cfg.warmup_ratio
weight_decay = cfg.weight_decay
grad_clipping = cfg.grad_clipping

scheduler = cfg.scheduler

use_tiktoken = cfg.use_tiktoken
seed = cfg.seed
