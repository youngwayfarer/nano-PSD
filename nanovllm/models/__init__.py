from nanovllm.models.qwen3 import Qwen3ForCausalLM

# ``model_type`` -> implementation class. qwen2 and qwen3 share one implementation;
# their differences (QKV bias, QK-norm) are handled by the explicit switches in
# ``nanovllm.models.qwen3.Qwen3Attention``.
MODEL_REGISTRY: dict[str, type] = {
    "qwen2": Qwen3ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}


def get_model_class(model_type: str):
    try:
        return MODEL_REGISTRY[model_type]
    except KeyError:
        supported = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(
            f"unsupported model_type {model_type!r}; supported: {supported}"
        ) from None
