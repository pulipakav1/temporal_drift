import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


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
    load_kwargs = {"torch_dtype": dtype, "num_labels": num_labels}
    if materialize_for_peft:
        # Avoid meta/offload hooks that can break PEFT injection.
        load_kwargs["device_map"] = None
        load_kwargs["low_cpu_mem_usage"] = False
    else:
        load_kwargs["device_map"] = device_map
    model_source = checkpoint_dir if checkpoint_dir and Path(checkpoint_dir).exists() else model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_source, **load_kwargs)
    tok = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok
