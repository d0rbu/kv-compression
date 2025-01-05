from typing import Any, Type

import arguably
from loguru import logger
from transformers import TrainingArguments

from data import TEXT_DATA
from core.model import get_model_and_tokenizer
from core.optimization.base import OptimizedRepresentation
from exp.util import OptimizationStrategy, optimization_strategies


@arguably.command
def main(
    model: str = "meta-llama/Llama-3.2-1B",
    strategy: OptimizationStrategy = OptimizationStrategy.kv,
    *args: Any,
    progressive: bool = False,
    data: str = "sorcerers_stone_4",
):
    assert strategy == OptimizationStrategy.kv, "only kv optimization is supported"
    assert not progressive, "progressive coding optimization is not supported yet"

    logger.info(f"using strategy: {strategy}")
    strategy: Type[OptimizedRepresentation] = optimization_strategies[strategy]

    logger.info(f"loading model and tokenizer from {model}")
    model, tokenizer = get_model_and_tokenizer(model)

    sample = TEXT_DATA[data]

    logger.info(f"compressing and decompressing sample of length {len(sample)}")

    compressed, compression_metadata = strategy.compress(
        sample,
        model,
        tokenizer,
        num_tokens=len(sample) // 65_536 or 1,
        training_args=TrainingArguments(
            output_dir="tmp",
            overwrite_output_dir=True,
            per_device_train_batch_size=1,
            num_train_epochs=len(sample) // 4 or 1,
            weight_decay=0.0,
            learning_rate=2e-3,
            logging_steps=16,
            logging_first_step=True,
            max_grad_norm=1.0,
        ),
    )

    logger.info(compression_metadata)

    decompressed, decompression_metadata = strategy.decompress(compressed, model, tokenizer)

    logger.info(decompression_metadata)
    logger.info(compressed)
    logger.info(decompressed[:1000])


if __name__ == "__main__":
    arguably.run()
