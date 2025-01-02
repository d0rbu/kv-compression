from typing import Any, Type

import arguably
from loguru import logger

from core.model import get_model_and_tokenizer
from exp.util import OptimizationStrategy, optimization_strategies


@arguably.command
def main(
    model: str = "meta-llama/Llama-3.2-3B",
    strategy: OptimizationStrategy = OptimizationStrategy.kv,
    *args: Any,
    progressive: bool = False,
):
    assert strategy == OptimizationStrategy.kv, "only kv optimization is supported"
    assert not progressive, "progressive coding optimization is not supported yet"

    logger.info(f"loading model and tokenizer from {model}")
    model, tokenizer = get_model_and_tokenizer()

    strategy: Type[OptimizationStrategy] = optimization_strategies[strategy]
    logger.info(model)
    logger.info(tokenizer)
    logger.info(strategy)


if __name__ == "__main__":
    arguably.run()
