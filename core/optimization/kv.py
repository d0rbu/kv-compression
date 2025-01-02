import types
from functools import partial
from typing import Any, Mapping, Self, Type

import torch as th
import torch.nn as nn
from loguru import logger
from torch.utils.data import TensorDataset
from transformers import (
    AdamW,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.cache_utils import DynamicCache

from core.optimization.base import OptimizedRepresentation

Data = str
Metadata = Mapping[str, Any]
CompressedData = DynamicCache
Model = PreTrainedModel


class KVRepresentation(OptimizedRepresentation):
    @classmethod
    def _compress(
        cls: Type[Self],
        data: Data,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        num_tokens: int = 128,
        training_args: TrainingArguments = TrainingArguments(),
    ) -> tuple[CompressedData, Metadata]:
        # use hf trainer with standard sequence modelling objective
        input_ids = tokenizer(data, return_tensors="pt").input_ids
        dataset = TensorDataset(input_ids)

        # we want to optimize the kv cache, the rest of the model is not important
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        num_kv_heads = model.config.num_key_value_heads
        num_layers = model.config.num_hidden_layers

        # legacy cache format is shape (l, 2, b, h, t, d') where d' = head_dim
        # and the first two dims are tuples while the rest are a tensor
        kv_tokens = nn.Parameter(
            th.empty(size=(num_layers, 2, 1, num_kv_heads, num_tokens, head_dim))
        )
        nn.init.xavier_uniform_(kv_tokens)

        kv_cache = DynamicCache.from_legacy_cache(kv_tokens)

        # monkey patch the forward method to use the kv cache during training
        original_forward = model.forward

        def forward_with_kv_cache(*args: Any, **kwargs: Any) -> Any:
            return original_forward(*args, **kwargs, past_key_values=kv_cache)

        model.forward = types.MethodType(forward_with_kv_cache, model)

        optim = AdamW(
            (kv_tokens,),
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            optimizers=(optim,),
        )
        train_output = trainer.train()._asdict()
        train_output["size"] = kv_tokens.numel()
        train_output["bytes"] = kv_tokens.element_size() * kv_tokens.numel()

        # restore the original forward method
        model.forward = original_forward

        return kv_cache, train_output

    @classmethod
    def _decompress(
        cls: Type[Self],
        compressed_data: CompressedData,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        generation_config: GenerationConfig = GenerationConfig(),
    ) -> tuple[Data, Metadata]:
        generate = partial(model.generate, past_key_values=compressed_data)

        if not generation_config.return_dict_in_generate:
            logger.warning(
                "generation_config.return_dict_in_generate is False; setting it to True"
            )
            generation_config.return_dict_in_generate = True

        if generation_config.num_return_sequences != 1:
            logger.warning(
                "generation_config.num_return_sequences is not 1; setting it to 1"
            )
            generation_config.num_return_sequences = 1

        output = generate(generation_config=generation_config)

        text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

        return text, output
