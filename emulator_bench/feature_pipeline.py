import gc
import os
import re
import sys
import zipfile
import zlib
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import T5EncoderModel, T5Tokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from utils import split

from emulator_bench.common import normalize_sequence, normalize_smiles


PROT_MODEL_ID = "Rostlab/prot_t5_xl_uniref50"
SMILES_VECTOR_DIM = 1024
PROTEIN_VECTOR_DIM = 1024


def resolve_amp_dtype(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None, "fp32"
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(index)
    if major >= 8:
        return torch.bfloat16, "bf16-mixed"
    return torch.float16, "fp16-mixed"


def autocast_context(device: torch.device, dtype):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def enable_cuda_fast_math() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def unikp_sequence_text(sequence: str) -> str:
    sequence = normalize_sequence(sequence)
    if len(sequence) > 1000:
        sequence = sequence[:500] + sequence[-500:]
    sequence = re.sub(r"[UZOB]", "X", sequence)
    return " ".join(list(sequence))


def load_prot_t5(device: torch.device):
    tokenizer = T5Tokenizer.from_pretrained(PROT_MODEL_ID, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(PROT_MODEL_ID)
    model = model.to(device)
    model.eval()
    gc.collect()
    return tokenizer, model


def embed_protein_sequences(
    sequences: Sequence[str],
    device: torch.device,
    batch_size: int = 4,
    max_tokens: int = 4096,
) -> Dict[str, np.ndarray]:
    if not sequences:
        return {}
    enable_cuda_fast_math()
    tokenizer, model = load_prot_t5(device)
    amp_dtype, precision = resolve_amp_dtype(device)
    print(f"ProtT5 device: {device} | precision: {precision}", flush=True)

    unique = sorted({normalize_sequence(seq) for seq in sequences}, key=len, reverse=True)
    result: Dict[str, np.ndarray] = {}
    batches: List[List[str]] = []
    current: List[str] = []
    current_tokens = 0
    for sequence in unique:
        token_count = len(sequence) + 1
        if current and (len(current) >= batch_size or current_tokens + token_count > max_tokens):
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(sequence)
        current_tokens += token_count
    if current:
        batches.append(current)

    for batch in tqdm(batches, desc="ProtT5 embedding", unit="batch"):
        formatted = [unikp_sequence_text(sequence) for sequence in batch]
        encoded = tokenizer.batch_encode_plus(formatted, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(encoded["input_ids"], device=device)
        attention_mask = torch.tensor(encoded["attention_mask"], device=device)
        with torch.no_grad(), autocast_context(device, amp_dtype):
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = hidden.detach().float().cpu()
        mask = attention_mask.detach().cpu()
        for idx, sequence in enumerate(batch):
            seq_len = int((mask[idx] == 1).sum().item())
            vector = hidden[idx, : seq_len - 1].mean(dim=0).numpy().astype(np.float32, copy=False)
            result[sequence] = vector
    return result


def load_smiles_transformer(device: torch.device):
    import __main__

    # The repository vocab.pkl was pickled from a script context, so pickle
    # expects WordVocab to be reachable from __main__ when loaded.
    if not hasattr(__main__, "WordVocab"):
        setattr(__main__, "WordVocab", WordVocab)
    vocab = WordVocab.load_vocab(str(REPO_ROOT / "vocab.pkl"))
    model = TrfmSeq2seq(len(vocab), 256, len(vocab), 4)
    state = torch.load(str(REPO_ROOT / "trfm_12_23000.pkl"), map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return vocab, model


def smiles_to_ids(smiles: str, vocab) -> List[int]:
    pad_index = 0
    unk_index = 1
    eos_index = 2
    sos_index = 3
    tokens = split(normalize_smiles(smiles)).split()
    if len(tokens) > 218:
        tokens = tokens[:109] + tokens[-109:]
    ids = [sos_index] + [vocab.stoi.get(token, unk_index) for token in tokens] + [eos_index]
    ids.extend([pad_index] * (220 - len(ids)))
    return ids


def _encode_smiles_batch(model, src: torch.Tensor, device: torch.device, amp_dtype) -> np.ndarray:
    with torch.no_grad(), autocast_context(device, amp_dtype):
        embedded = model.embed(src)
        embedded = model.pe(embedded)
        output = embedded
        for idx in range(model.trfm.encoder.num_layers - 1):
            output = model.trfm.encoder.layers[idx](output, src_mask=None)
        penul = output
        output = model.trfm.encoder.layers[-1](output, src_mask=None)
        if model.trfm.encoder.norm:
            output = model.trfm.encoder.norm(output)
        output_f = output.detach().float().cpu().numpy()
        penul_f = penul.detach().float().cpu().numpy()
    return np.hstack(
        [
            np.mean(output_f, axis=0),
            np.max(output_f, axis=0),
            output_f[0, :, :],
            penul_f[0, :, :],
        ]
    ).astype(np.float32, copy=False)


def embed_smiles_values(
    smiles_values: Sequence[str],
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, np.ndarray]:
    if not smiles_values:
        return {}
    enable_cuda_fast_math()
    vocab, model = load_smiles_transformer(device)
    amp_dtype, precision = resolve_amp_dtype(device)
    print(f"SMILES Transformer device: {device} | precision: {precision}", flush=True)

    unique = sorted({normalize_smiles(smiles) for smiles in smiles_values}, key=len, reverse=True)
    result: Dict[str, np.ndarray] = {}
    for start in tqdm(range(0, len(unique), batch_size), desc="SMILES Transformer embedding", unit="batch"):
        batch = unique[start : start + batch_size]
        ids = [smiles_to_ids(smiles, vocab) for smiles in batch]
        src = torch.tensor(ids, dtype=torch.long, device=device).t().contiguous()
        vectors = _encode_smiles_batch(model, src, device=device, amp_dtype=amp_dtype)
        for idx, smiles in enumerate(batch):
            result[smiles] = vectors[idx]
    return result


def save_npz_atomic(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with open(tmp_path, "wb") as handle:
        np.savez_compressed(handle, **payload)
    tmp_path.replace(path)


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except (zipfile.BadZipFile, EOFError, OSError, ValueError, zlib.error) as exc:
        raise RuntimeError(f"Corrupted cache file: {path}. Rebuild it with --overwrite.") from exc


def load_cached_vector(path: Path, key: str = "vector") -> np.ndarray:
    payload = load_npz(path)
    if key not in payload:
        raise KeyError(f"Missing `{key}` in cache file {path}")
    return payload[key].astype(np.float32, copy=False)


def iter_unique_from_frames(frames: Iterable, sequence_col: str, smiles_col: str):
    sequences = set()
    smiles_values = set()
    for frame in frames:
        sequences.update(normalize_sequence(value) for value in frame[sequence_col].astype(str))
        smiles_values.update(normalize_smiles(value) for value in frame[smiles_col].astype(str))
    return sorted(sequences), sorted(smiles_values)
