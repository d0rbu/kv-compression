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
from peft import LoraConfig

from core.optimization.base import OptimizedRepresentation
from core.optimization.util import fix_quantization_class, monkey_patch_trainable_bos_token, update_dict_like, TRUE_DEFAULT_GENERATION_CONFIG, TRUE_DEFAULT_TRAINING_ARGS, TRUE_DEFAULT_LORA_CONFIG


Data = str
Metadata = Mapping[str, Any]
CompressedData = str
Model = PreTrainedModel


class LoRARepresentation(OptimizedRepresentation):
    DEFAULT_TRAINING_ARGS = TrainingArguments(
        output_dir="tmp",
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        num_train_epochs=1024,
        weight_decay=0.0,
        learning_rate=4e-4,
        logging_steps=32,
        logging_first_step=True,
        max_grad_norm=1.0,
    )

    DEFAULT_LORA_CONFIG = LoraConfig(
        r=1,
        target_modules=["v_proj"],
        lora_alpha=1,
        lora_dropout=0.0,
        layers_to_transform=[i for i in range(9, 16)],
        init_lora_weights="pissa",
    )

    @classmethod
    def _compress(
        cls: Type[Self],
        data: Data,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        r: int = 1,
        training_args: TrainingArguments = DEFAULT_TRAINING_ARGS,
        lora_config: LoraConfig = DEFAULT_LORA_CONFIG,
        **kwargs: Any,
    ) -> tuple[CompressedData, Metadata]:
        assert r > 0, "r must be greater than 0"

        # update default training args with user provided training args
        training_args = update_dict_like(
            training_args,
            cls.DEFAULT_TRAINING_ARGS,
            TRUE_DEFAULT_TRAINING_ARGS
        )
        # same for lora config
        lora_config = update_dict_like(
            lora_config,
            cls.DEFAULT_LORA_CONFIG,
            TRUE_DEFAULT_LORA_CONFIG
        )

        fix_quantization_class(model)

        adapter_name = "compression_adapter_0"
        model.add_adapter(lora_config, adapter_name=adapter_name)

        # use hf trainer with standard sequence modelling objective
        data_with_eos_token = data + tokenizer.eos_token
        input_ids = tokenizer(data_with_eos_token, return_tensors="pt").input_ids.squeeze(0)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=[{"input_ids": input_ids, "labels": input_ids}],
        )
        train_output = trainer.train()._asdict()

        adapter_weights = model.get_adapter_state_dict("compression_adapter_0")
        adapter_weight_sizes = [weight.numel() for weight in adapter_weights.values()]
        adapter_weight_byteses = [weight.numel() * weight.element_size() for weight in adapter_weights.values()]

        train_output["size"] = sum(adapter_weight_sizes)
        train_output["bytes"] = sum(adapter_weight_byteses)

        return adapter_name, train_output
    
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

        # enable the adapter
        model.set_adapter(compressed_data)
        model.enable_adapters()

        output = model.generate(
            generation_config=generation_config
        )

        text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

        return text, output
