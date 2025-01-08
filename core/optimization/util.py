
import types
import torch as th
from typing import Self, Sequence, Any, Callable, TypeVar
from functools import wraps
from transformers import PreTrainedModel, DynamicCache, TrainingArguments, GenerationConfig
from transformers.quantizers.quantizer_bnb_8bit import Bnb8BitHfQuantizer
from peft import LoraConfig


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


def monkey_patch_trainable_bos_token(
    model: PreTrainedModel,
    prefix_tokens: th.Tensor,
) -> Callable[[], None]:
    # monkey patch the forward method to prepend the input embeddings with the bos token
    # we also need to keep the signature (⁉️) because its used by hf trainer
    original_forward = model.forward
    original_unbound_forward = original_forward.__func__

    @wraps(original_unbound_forward)
    def unbound_forward_with_prefix(*args: Any, **kwargs: Any) -> Any:
        # we need to reinitialize the input embeds every forward pass
        # because .backward() clears the computation graph
        input_embeds = kwargs.pop("inputs_embeds", None)

        if input_embeds is None:
            raise ValueError("inputs_embeds must be provided to the monkeypatched model.forward method")

        labels = kwargs.pop("labels", None)
        if labels is None:
            raise ValueError("labels must be provided to the monkeypatched model.forward method")

        input_embeds = th.cat([prefix_tokens.unsqueeze(0), input_embeds], dim=1)
        ignored_labels = th.full((1, prefix_tokens.size(0)), fill_value=-100, dtype=th.long, device=labels.device)
        labels = th.cat([ignored_labels, labels], dim=1)

        return original_unbound_forward(*args, **kwargs, inputs_embeds=input_embeds, labels=labels)

    model.forward = types.MethodType(unbound_forward_with_prefix, model)

    def unpatch() -> None:
        model.forward = original_forward

    return unpatch


T = TypeVar("T")

def update_dict_like(target: T, source: T, default: T) -> T:
    # for things like hf generation config or training args
    target_dict = target.to_dict()
    source_dict = source.to_dict()
    default_dict = default.to_dict()

    non_default_source = {
        key: value
        for key, value in target_dict.items()
        if value != default_dict[key]
    }

    source_dict.update(non_default_source)
    
    dict_like = type(target)

    return dict_like(**source_dict)

TRUE_DEFAULT_TRAINING_ARGS = TrainingArguments(output_dir="tmp")
TRUE_DEFAULT_GENERATION_CONFIG = GenerationConfig()
TRUE_DEFAULT_LORA_CONFIG = LoraConfig()
