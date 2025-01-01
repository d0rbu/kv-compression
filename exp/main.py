import arguably
from typing import Any
from loguru import logger

from exp.util import OptimizationStrategy
from core.model import get_model_and_tokenizer


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
    logger.info(model)
    logger.info(tokenizer)


if __name__ == "__main__":
    arguably.run()
