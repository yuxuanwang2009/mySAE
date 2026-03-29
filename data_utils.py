from typing import Iterable

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import config
import tiktoken
from datasets import load_dataset

# -----------------------
# Tokenizer
# -----------------------
if config.use_tiktoken is True:
    tok = tiktoken.get_encoding("gpt2")
else:
    import regex_tokenizer as rt
    import tokenizer_utils as tu
    if config.vocab_size < 50256:
        tok = rt.RegexTokenizer.load("tokenizer.json")
        # use our custom tokenizer but GPT-2 merges/vocab, less than 2 times slower.
    else:
        tok = rt.RegexTokenizer(merges = tu.merges,
                                vocab = tu.vocab, 
                                special_token_to_id = {"<|endoftext|>":50256},
                                byte_shuffle = tu.byte_shuffle
                               )


# -----------------------
# Tokenize helpers
# -----------------------
def stot(s: str) -> torch.Tensor:
    ids = tok.encode(s, allowed_special={"<|endoftext|>"})
    return torch.tensor(ids, dtype=torch.long)


def ttos(t: torch.Tensor, for_output: bool = False) -> str:
    out = tok.decode(t.tolist())
    return out.replace("<|endoftext|>", "\n==========\n") if for_output else out


# -----------------------
# FineWeb-Edu (HF)
# -----------------------
DATASET_PATH = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-100BT"
DATASET_SUBDIR = "sample/100BT"  # where the parquet shards live
TRAIN_EPOCH_GROUPS = 13
TRAIN_FILE_RANGE = range(0, 8)  # 00000..00007


def _train_files_for_epoch(epoch: int, rank: int) -> list[str]:
    epoch_idx = int(epoch) % TRAIN_EPOCH_GROUPS
    files = [f"{DATASET_SUBDIR}/{epoch_idx:03d}_{rank:05d}.parquet"]
    return files


def _val_files() -> list[str]:
    files = [f"{DATASET_SUBDIR}/013_00000.parquet"]
    return files


# -----------------------
# Doc streams
# -----------------------
class HFDocStream(IterableDataset):
    """Stream docs from HF, sharded per rank+worker, with an optional doc cap."""

    def __init__(
        self,
        split: str,
        rank: int,
        world_size: int,
        limit: int | None = None,
        data_files=None,
    ):
        super().__init__()
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.limit = limit
        self.epoch = 0
        self.base_seed = int(config.seed)
        self.data_files = data_files

    def __iter__(self) -> Iterable[str]:
        # data_files is either a function or a list, but after this line it is a list of file names.
        data_files = self.data_files(self.epoch, self.rank) if callable(self.data_files) else self.data_files
        ds = load_dataset(
            DATASET_PATH,
            name=DATASET_CONFIG,
            split=self.split,
            streaming=True,
            data_files=data_files,
        )
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        seed = self.base_seed + self.epoch * 1000 + self.rank * 10 + worker_id
        ds = ds.shuffle(buffer_size=100_000, seed=seed)

        count = 0
        for row in ds:
            text = row.get("text", "") if isinstance(row, dict) else ""
            if text:
                yield text
                count += 1
                if self.limit is not None and count >= self.limit:
                    break

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)


class CachedHFDocStream(IterableDataset):
    """Stream a fixed cached slice from HF, broadcast to all ranks, then shard locally."""

    def __init__(self, split: str, rank: int, world_size: int, limit: int, data_files=None):
        super().__init__()
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.limit = limit
        self.data_files = data_files
        self._cached_docs = None

    def _ensure_cache(self):
        if self._cached_docs is not None:
            return

        docs = []
        distributed = dist.is_available() and dist.is_initialized()

        def _pull():
            ds = load_dataset(
                DATASET_PATH,
                name=DATASET_CONFIG,
                split=self.split,
                streaming=True,
                data_files=self.data_files,
            )
            for row in ds:
                text = row.get("text", "") if isinstance(row, dict) else ""
                if text:
                    docs.append(text)
                if len(docs) >= self.limit:
                    break

        if distributed:
            if self.rank == 0:
                _pull()
            obj = [docs]
            dist.broadcast_object_list(obj, src=0)
            docs = obj[0]
        else:
            _pull()

        # Make evenly splittable across ranks for deterministic per-rank slices
        if distributed and docs:
            total = len(docs) - (len(docs) % self.world_size)
            docs = docs[:total] if total > 0 else docs

        self._cached_docs = docs

    def __iter__(self) -> Iterable[str]:
        self._ensure_cache()
        total = len(self._cached_docs)
        start = (total * self.rank) // self.world_size
        end = (total * (self.rank + 1)) // self.world_size
        for doc in self._cached_docs[start:end]:
            yield doc

    def __len__(self):
        self._ensure_cache()
        total = len(self._cached_docs)
        start = (total * self.rank) // self.world_size
        end = (total * (self.rank + 1)) // self.world_size
        return end - start


# -----------------------
# Block stream
# -----------------------
class BlockStream(IterableDataset):
    """Stream fixed-length (x, y) blocks from a doc iterable (the DocStreams)."""

    def __init__(self, doc_iterable: IterableDataset, materialize: bool = False):
        super().__init__()
        self.doc_iterable = doc_iterable
        self.materialize = materialize
        self._blocks = None
        if self.materialize:
            self._blocks = list(self._iter_blocks())

    def _iter_blocks(self):
        T = config.T

        # Use EOT as pad: avoids multi-token newline issues + is standard for GPT-style LM
        eot_tokens = stot("<|endoftext|>")
        if eot_tokens.numel() != 1:
            raise ValueError(
                "Tokenizer does not have a single <|endoftext|> token. "
                "Please train your tokenizer with special_tokens=['<|endoftext|>'] or switch to tiktoken."
            )
        eot_id = eot_tokens.item()
        pad_id = eot_id

        buf: list[int] = []
        for doc in self.doc_iterable:
            ids = tok.encode(doc, allowed_special={"<|endoftext|>"})
            ids.append(eot_id)
            buf.extend(ids)
            # keep the first T tokens and leave the remainder to combine with tokens in next doc
            while len(buf) >= T + 1:
                block = buf[: T + 1]
                buf = buf[T + 1 :]
                block_t = torch.tensor(block, dtype=torch.long)
                yield block_t[:-1], block_t[1:]

        # pad the leftover tokens in buf
        if buf:
            if len(buf) < T + 1:
                buf = buf + [pad_id] * (T + 1 - len(buf))
            block_t = torch.tensor(buf[: T + 1], dtype=torch.long)
            yield block_t[:-1], block_t[1:]

    def __iter__(self):
        if self.materialize and self._blocks is not None:
            return iter(self._blocks)
        return self._iter_blocks()

    def __len__(self):
        if self.materialize and self._blocks is not None:
            return len(self._blocks)
        raise TypeError("Length not available for streaming BlockStream")

    def set_epoch(self, epoch: int):
        if hasattr(self.doc_iterable, "set_epoch"):
            self.doc_iterable.set_epoch(epoch)


# -----------------------
# Builders
# -----------------------
def Build_datasets(rank: int = 0, world_size: int = 1):
    """Create streaming train/val IterableDatasets. FineWeb-Edu only exposes 'train'."""
    train_limit = 1_000 if config.device.type != "cuda" else None

    train_ds = HFDocStream("train", rank, world_size, limit=train_limit, data_files=_train_files_for_epoch)

    val_world = world_size if (dist.is_available() and dist.is_initialized()) else 1
    val_limit = 50 if config.device.type != "cuda" else 12_000
    val_ds = CachedHFDocStream("train", rank, val_world, limit=val_limit, data_files=_val_files())

    return train_ds, val_ds


def Construct_data_loaders(data) -> DataLoader:
    """Wrap train/val doc streams into block DataLoaders (batch_size = blocks)."""
    if isinstance(data, tuple) and len(data) == 2 and all(isinstance(d, IterableDataset) for d in data):
        ds_tr, ds_va = data
    else:
        raise TypeError(f"data must be a (train_ds, val_ds) tuple, got {type(data)}")

    # Wrap doc streams into block streams (train stays streaming, val may be materialized)
    bs_tr = BlockStream(ds_tr, materialize=False)
    materialize_val = isinstance(ds_va, CachedHFDocStream)
    bs_va = BlockStream(ds_va, materialize=materialize_val)

    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0

    # Keep your num_workers logic; just fix syntax
    train_loader = DataLoader(
        bs_tr,
        batch_size=config.batch_size,
        # HF streaming can't be split across more workers than shards; keep it at 1 per process
        num_workers=1 if dist.is_initialized() else 1 if config.device.type == "mps" else 0,
        pin_memory=True if dist.is_available() and dist.is_initialized() else False,
        prefetch_factor=2 if dist.is_available() and dist.is_initialized() else 2 if config.device.type == "mps" else None,
        persistent_workers=True if dist.is_available() and dist.is_initialized() else True if config.device.type == "mps" else False,
    )
    val_loader = DataLoader(
        bs_va,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
    )

    if rank == 0:
        print("Training set: streaming blocks (length unknown)", flush=True)

    local_blocks = len(bs_va)
    total_blocks = local_blocks
    if distributed:
        blk_tensor = torch.tensor([local_blocks], device=config.device if config.device.type == "cuda" else "cpu")
        dist.reduce(blk_tensor, dst=0, op=dist.ReduceOp.SUM)
        if rank == 0:
            total_blocks = int(blk_tensor.item())
    if rank == 0:
        total_batches = (total_blocks + config.batch_size - 1) // config.batch_size
        print(f"Validation set: {total_blocks} blocks, loader batches: {total_batches}", flush=True)

    return train_loader, val_loader
