from typing import Any, Mapping, Self, Type

import torch as th
import torch.nn as nn
from loguru import logger
from transformers import (
    AdamW,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)
from galore_torch import GaLoreAdamW, GaLoreAdamW8bit
from transformers.cache_utils import DynamicCache

from core.optimization.base import OptimizedRepresentation
from core.optimization.util import fix_quantization_class, monkey_patch_kv_cache, update_dict_like, TRUE_DEFAULT_TRAINING_ARGS, TRUE_DEFAULT_GENERATION_CONFIG


Data = str
Metadata = Mapping[str, Any]
CompressedData = DynamicCache
Model = PreTrainedModel

class KVRepresentation(OptimizedRepresentation):
    DEFAULT_TRAINING_ARGS = TrainingArguments(
        output_dir="tmp",
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        num_train_epochs=1024,
        weight_decay=0.0,
        learning_rate=2e-3,
        logging_steps=16,
        logging_first_step=True,
        max_grad_norm=1.0,
    )
    COMPRESSED_PATH = "kv_tokens.pt"


    @classmethod
    def _compress(
        cls: Type[Self],
        data: Data,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        num_tokens: int = 128,
        training_args: TrainingArguments = DEFAULT_TRAINING_ARGS,
        load: bool = False,
    ) -> tuple[CompressedData, Metadata]:
        assert num_tokens > 0, "num_tokens must be greater than 0"

        # update default training args with user provided training args
        training_args = update_dict_like(
            training_args,
            cls.DEFAULT_TRAINING_ARGS,
            TRUE_DEFAULT_TRAINING_ARGS
        )

        fix_quantization_class(model)

        # use hf trainer with standard sequence modelling objective
        data_with_eos_token = data + tokenizer.eos_token
        input_ids = tokenizer(data_with_eos_token, return_tensors="pt").input_ids.squeeze(0)

        # we want to optimize the kv cache, the rest of the model is not important
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        num_kv_heads = getattr(model.config, "num_key_value_heads", 0) or model.config.num_attention_heads
        num_layers = model.config.num_hidden_layers

        # legacy cache format is shape (l, 2, b, h, t, d') where d' = head_dim
        # and the first two dims are tuples while the rest are a tensor
        if load:
            kv_tokens = th.load(cls.COMPRESSED_PATH)
        else:
            kv_tokens = th.empty(
                size=(num_layers, 2, 1, num_kv_heads, num_tokens, head_dim),
                dtype=th.float32,
                requires_grad=True,
                device=model.device,
            )
            nn.init.xavier_uniform_(kv_tokens)

        unpatch = monkey_patch_kv_cache(model, kv_tokens)

        optim = AdamW(
            [kv_tokens],
            lr=training_args.learning_rate,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            weight_decay=training_args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optim,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.num_train_epochs,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=[{"input_ids": input_ids, "labels": input_ids}],
            optimizers=(optim, scheduler),
        )
        train_output = trainer.train()._asdict()
        train_output["size"] = kv_tokens.numel()
        train_output["bytes"] = kv_tokens.element_size() * kv_tokens.numel()

        # restore the original forward method
        unpatch()

        # save to disk
        th.save(kv_tokens, cls.COMPRESSED_PATH)

        return DynamicCache.from_legacy_cache(kv_tokens), train_output
    
    DEFAULT_GENERATION_CONFIG = GenerationConfig(
        max_length=131072,
        num_return_sequences=1,
        return_dict_in_generate=True,
        do_sample=False,
        output_logits=True,
    )

    @classmethod
    def _decompress(
        cls: Type[Self],
        compressed_data: CompressedData,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        generation_config: GenerationConfig = DEFAULT_GENERATION_CONFIG
    ) -> tuple[Data, Metadata]:
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
        
        # update default generation config with user provided generation config
        generation_config = update_dict_like(
            generation_config,
            cls.DEFAULT_GENERATION_CONFIG,
            TRUE_DEFAULT_GENERATION_CONFIG
        )

        # model.generate assumes our input ids includes the cached tokens, so we prefill with random values
        input_ids = th.empty(
            size=(1, compressed_data.get_seq_length() + 1),
            dtype=th.long,
            device=model.device,
        )
        input_ids[-1] = tokenizer.bos_token_id

        output = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            past_key_values=compressed_data
        )

        text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

        return text, output
