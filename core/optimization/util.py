
import types
import torch as th
from typing import Self, Sequence, Any, Callable
from functools import wraps
from transformers import PreTrainedModel, DynamicCache
from transformers.quantizers.quantizer_bnb_8bit import Bnb8BitHfQuantizer


class Modified8BitHfQuantizer(Bnb8BitHfQuantizer):
    @property
    def is_qat_trainable(self: Self) -> bool:
        return True

quantization_replacement = {
    Bnb8BitHfQuantizer: Modified8BitHfQuantizer,
}

def fix_quantization_class(model: PreTrainedModel) -> None:
    quantizer = getattr(model, "hf_quantizer", None)

    if quantizer is None:
        return

    quantization_classes = [
        replacement_class
        for original_class, replacement_class in quantization_replacement.items()
        if isinstance(model.hf_quantizer, original_class)
    ]
    if len(quantization_classes) > 0:
        model.hf_quantizer = quantization_classes[0](
            quantization_config=model.hf_quantizer.quantization_config
        )


def monkey_patch_kv_cache(
    model: PreTrainedModel,
    kv_tokens: Sequence[Sequence[th.Tensor]],
) -> Callable[[], None]:
    # monkey patch the forward method to use the kv cache during training
    # we also need to keep the signature (⁉️) because its used by hf trainer
    original_forward = model.forward
    original_unbound_forward = original_forward.__func__

    @wraps(original_unbound_forward)
    def unbound_forward_with_kv_cache(*args: Any, **kwargs: Any) -> Any:
        # we need to reinitialize the cache every forward pass
        # because .backward() clears the computation graph
        kv_cache = DynamicCache.from_legacy_cache(kv_tokens)

        return original_unbound_forward(*args, **kwargs, past_key_values=kv_cache)

    model.forward = types.MethodType(unbound_forward_with_kv_cache, model)

    def unpatch() -> None:
        model.forward = original_forward

    return unpatch
