import json
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _valid_checkpoint(path: str) -> bool:
    cfg_file = Path(path) / "config.json"
    if not cfg_file.exists():
        return False
    try:
        with cfg_file.open() as f:
            return "model_type" in json.load(f)
    except Exception:
        return False


def load_model_tokenizer(
    model_name: str,
    num_labels: int,
    bf16: bool = True,
    device_map: str = "auto",
    materialize_for_peft: bool = True,
    checkpoint_dir: str | None = None,
):
    has_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if (bf16 and has_cuda) else torch.float32
    load_kwargs = {"dtype": dtype, "num_labels": num_labels}
    if materialize_for_peft:
        # Avoid meta/offload hooks that can break PEFT injection.
        load_kwargs["device_map"] = None
        load_kwargs["low_cpu_mem_usage"] = False
    else:
        load_kwargs["device_map"] = device_map
    ckpt_valid = checkpoint_dir and Path(checkpoint_dir).exists() and _valid_checkpoint(checkpoint_dir)
    if checkpoint_dir and Path(checkpoint_dir).exists() and not ckpt_valid:
        print(f"[base_model] WARNING: checkpoint at {checkpoint_dir} is invalid/incomplete — retraining from base model")
    model_source = checkpoint_dir if ckpt_valid else model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_source, **load_kwargs)
    tok = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, tok
