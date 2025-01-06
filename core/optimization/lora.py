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

from core.optimization.base import OptimizedRepresentation
from core.optimization.util import fix_quantization_class, monkey_patch_trainable_bos_token, update_dict_like, TRUE_DEFAULT_GENERATION_CONFIG, TRUE_DEFAULT_TRAINING_ARGS


Data = str
Metadata = Mapping[str, Any]
CompressedData = th.Tensor
Model = PreTrainedModel


class LoRARepresentation(OptimizedRepresentation):
    DEFAULT_TRAINING_ARGS = TrainingArguments(
        output_dir="tmp",
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        num_train_epochs=1024,
        weight_decay=0.0,
        learning_rate=5e-5,
        logging_steps=32,
        logging_first_step=True,
        max_grad_norm=1.0,
    )


    @classmethod
    def _compress(
        cls: Type[Self],
        data: Data,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        r: int = 1,
        training_args: TrainingArguments = DEFAULT_TRAINING_ARGS,
    ) -> tuple[CompressedData, Metadata]:
        assert r > 0, "r must be greater than 0"

        # update default training args with user provided training args
        training_args = update_dict_like(
            training_args,
            cls.DEFAULT_TRAINING_ARGS,
            TRUE_DEFAULT_TRAINING_ARGS
        )

        fix_quantization_class(model)

        # TODO: add an adapter for lora

        # use hf trainer with standard sequence modelling objective
        data_with_eos_token = data + tokenizer.eos_token
        input_ids = tokenizer(data_with_eos_token, return_tensors="pt").input_ids.squeeze(0).to(model.device)
        embeddings = model.get_input_embeddings()
        inputs_embeds = embeddings(input_ids)

        prefix_tokens = th.empty(num_tokens, model.config.hidden_size, device=model.device, requires_grad=True)
        nn.init.xavier_uniform_(prefix_tokens)

        unpatch = monkey_patch_trainable_bos_token(model, prefix_tokens)

        optim = AdamW(
            [prefix_tokens],
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
            train_dataset=[{"inputs_embeds": inputs_embeds.detach(), "labels": input_ids}],
            optimizers=(optim, scheduler),
        )
        train_output = trainer.train()._asdict()
        train_output["size"] = prefix_tokens.numel()
        train_output["bytes"] = prefix_tokens.numel() * prefix_tokens.element_size()

        # restore the original forward method
        unpatch()

        return prefix_tokens, train_output
    
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

        output = model.generate(
            inputs_embeds=compressed_data.unsqueeze(0),
            generation_config=generation_config
        )

        text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

        return text, output
