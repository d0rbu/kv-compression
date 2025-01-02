import torch as th
from loguru import logger
from functools import partial
from torch.utils.data import TensorDataset
from transformers.cache_utils import DynamicCache
from typing import Mapping, Any, Callable, Self, Type
from transformers import PreTrainedModel, PreTrainedTokenizer, GenerationConfig, Trainer, TrainingArguments, AdamW

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
        # get model attention k dim
        kv_tokens = th.tensor([tokenizer.bos_token_id] * num_tokens).unsqueeze(0)
        forward = partial(model.forward, past_key_values=kv_tokens)
        optim = AdamW(model.parameters(), lr=1e-5)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            optimizers=
        )

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
            logger.warning("generation_config.return_dict_in_generate is False; setting it to True")
            generation_config.return_dict_in_generate = True

        if generation_config.num_return_sequences != 1:
            logger.warning("generation_config.num_return_sequences is not 1; setting it to 1")
            generation_config.num_return_sequences = 1

        output = generate(generation_config=generation_config)

        text = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]

        return text, output
