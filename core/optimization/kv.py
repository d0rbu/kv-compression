from torch import Tensor
from loguru import logger
from typing import Mapping, Any, Callable
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache

from core.optimization.base import OptimizedRepresentation


Data = str
Metadata = Mapping[str, Any]
CompressedData = DynamicCache
Model = PreTrainedModel


class KVRepresentation(OptimizedRepresentation):
    @staticmethod
    def _compress(
        data: Data,
        model: Model,
        tokenizer: PreTrainedTokenizer,
        num_tokens: int = 128,
    ) -> tuple[CompressedData, Metadata]:
        pass  # TODO: continuously optimize kv cache of size num_tokens on given model to generate the given data

    @staticmethod
    def _decompress(
        compressed_data: CompressedData,
        model: Model | None,
        tokenizer: PreTrainedTokenizer
    ) -> tuple[Data, Metadata]:
        pass  # TODO: decode the compressed data using the given model, tokenizer, and kv cache

    @staticmethod
    def hook_kv_cache(
        generate_method: Callable,
        kv_cache: DynamicCache,
    ) -> Callable:
        pass  # TODO: return a partial function that hooks the kv cache to the generate method
