from torch import Tensor
from loguru import logger
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Self, Sequence, Mapping, TypeVar, Type
from transformers import PreTrainedModel, PreTrainedTokenizer


Data = TypeVar("Data")
CompressedData = TypeVar("CompressedData")
Model = TypeVar("Model", PreTrainedModel, str)


class OptimizedRepresentation(ABC):
    @classmethod
    @abstractmethod
    def _compress(
        cls: Type[Self],
        data: Data,
        model: Model | None,
        tokenizer: PreTrainedTokenizer | None
    ) -> CompressedData:
        pass

    @classmethod
    @abstractmethod
    def _decompress(
        cls: Type[Self],
        compressed_data: CompressedData,
        model: Model | None,
        tokenizer: PreTrainedTokenizer | None
    ) -> Data:
        pass

    @classmethod
    @lru_cache
    def compress(
        cls: Type[Self],
        data: Data,
        model: Model | None,
        tokenizer: PreTrainedTokenizer | None
    ) -> CompressedData:
        return cls._compress(data, model, tokenizer)

    @classmethod
    @lru_cache
    def decompress(
        cls: Type[Self],
        compressed_data: CompressedData | None,
        model: Model | None,
        tokenizer: PreTrainedTokenizer | None
    ) -> Data:
        return cls._decompress(compressed_data, model, tokenizer)
