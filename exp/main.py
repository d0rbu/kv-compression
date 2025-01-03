from typing import Any, Type

import arguably
from loguru import logger

import data
from core.model import get_model_and_tokenizer
from core.optimization.base import OptimizedRepresentation
from exp.util import OptimizationStrategy, optimization_strategies


@arguably.command
def main(
    model: str = "meta-llama/Llama-3.2-1B",
    strategy: OptimizationStrategy = OptimizationStrategy.kv,
    *args: Any,
    progressive: bool = False,
):
    assert strategy == OptimizationStrategy.kv, "only kv optimization is supported"
    assert not progressive, "progressive coding optimization is not supported yet"

    logger.info(f"using strategy: {strategy}")
    strategy: Type[OptimizedRepresentation] = optimization_strategies[strategy]

    logger.info(f"loading model and tokenizer from {model}")
    model, tokenizer = get_model_and_tokenizer()

    sample = data.alice

    logger.info("compressing and decompressing sample")

    compressed, compression_metadata = strategy.compress(sample, model, tokenizer)

    logger.info(compression_metadata)

    decompressed, decompression_metadata = strategy.decompress(compressed, model, tokenizer)

    logger.info(decompression_metadata)
    logger.info(compressed[:1000])
    logger.info(decompressed[:1000])


if __name__ == "__main__":
    arguably.run()
