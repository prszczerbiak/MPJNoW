"""Wspólna konfiguracja ładowania modeli i kontrola wymagań kwantyzacji."""
from .common import ProtocolError

GENERATOR_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EVALUATOR_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def validate_loading(cfg):
    if cfg["precision"] not in ("bfloat16", "float16", "float32"):
        raise ProtocolError("Unsupported compute precision")
    quantization = cfg.get("quantization")
    if quantization not in (None, "4bit", "8bit"):
        raise ProtocolError("Quantization must be null, 4bit, or 8bit")
    if quantization and not cfg.get("quantization_justification"):
        raise ProtocolError("Quantization requires a documented hardware justification")
    if quantization == "4bit":
        if cfg.get("quantization_type", "nf4") not in ("nf4", "fp4"):
            raise ProtocolError("Invalid 4-bit quantization type")
        if type(cfg.get("double_quantization", True)) is not bool:
            raise ProtocolError("double_quantization must be a boolean")


def loading_kwargs(cfg):
    import torch
    from transformers import BitsAndBytesConfig
    validate_loading(cfg)
    dtype = getattr(torch, cfg["precision"])
    kwargs = {"torch_dtype": dtype, "device_map": cfg["device_map"],
              "attn_implementation": cfg.get("attn_implementation", "sdpa")}
    quantization = cfg.get("quantization")
    if quantization:
        # Profil projektu wymaga CUDA; środowisko CPU służy wyłącznie do testów.
        if not torch.cuda.is_available():
            raise ProtocolError("Quantized inference requires CUDA-enabled PyTorch and an exposed NVIDIA GPU; the CPU test container cannot run this profile")
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise ProtocolError("The GPU does not support bfloat16; amend and freeze the compute profile before inference")
        try:
            import importlib.metadata
            importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError as error:
            raise ProtocolError("Install bitsandbytes from requirements.txt before quantized inference") from error
        options = {"load_in_4bit": quantization == "4bit", "load_in_8bit": quantization == "8bit"}
        if quantization == "4bit":
            options.update(bnb_4bit_compute_dtype=dtype,
                           bnb_4bit_quant_type=cfg.get("quantization_type", "nf4"),
                           bnb_4bit_use_double_quant=cfg.get("double_quantization", True))
        kwargs["quantization_config"] = BitsAndBytesConfig(**options)
    return kwargs
