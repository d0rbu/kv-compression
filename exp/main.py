from typing import Any, Type

import sys
import arguably
from loguru import logger
from transformers import TrainingArguments, GenerationConfig

from data import TEXT_DATA
from core.model import get_model_and_tokenizer
from core.optimization.base import OptimizedRepresentation
from exp.util import OptimizationStrategy, optimization_strategies


@arguably.command
def main(
    *args: Any,
    model: str = "meta-llama/Llama-3.2-1B",
    strategy: OptimizationStrategy = OptimizationStrategy.kv,
    progressive: bool = False,
    data: str = "sorcerers_stone_4",
    log_level: str = "INFO",
    lr: float | None = None,
    epochs: int | None = None,
    load: bool = False,
):
    assert not progressive, "progressive coding optimization is not supported yet"

    logger.remove()
    logger.add(sys.stderr, level=log_level)

    logger.info(f"using strategy: {strategy}")
    strategy: Type[OptimizedRepresentation] = optimization_strategies[strategy]

    logger.info(f"loading model and tokenizer from {model}")
    model, tokenizer = get_model_and_tokenizer(model)

    sample = TEXT_DATA[data]

    logger.info(f"compressing and decompressing sample of length {len(sample)}")

    num_tokens = len(sample) // 65_536 or 1
    # num_tokens = 8
    compressed, compression_metadata = strategy._compress(
        sample,
        model,
        tokenizer,
        num_tokens=num_tokens,
        training_args=TrainingArguments(
            output_dir="tmp",
            num_train_epochs=epochs or len(sample) * 2 + 600,
            weight_decay=0.0,
            logging_steps=32,
            logging_first_step=True,
            max_grad_norm=1.0,
            learning_rate=lr or 0.00005,
        ),
        load=load,
    )

    logger.info(compression_metadata)

    decompressed, decompression_metadata = strategy.decompress(
        compressed,
        model,
        tokenizer,
        generation_config=GenerationConfig(
            max_length=len(sample) + num_tokens,
        ),
    )

    logger.debug(decompression_metadata)
    logger.info(compressed)
    logger.info(decompressed)


if __name__ == "__main__":
    arguably.run()
