from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Self, Type, TypeVar, Any, Mapping

from transformers import PreTrainedModel, PreTrainedTokenizer

Data = TypeVar("Data")
Metadata = Mapping[str, Any]
CompressedData = TypeVar("CompressedData")
Model = TypeVar("Model", PreTrainedModel, str)


class OptimizedRepresentation(ABC):
    @classmethod
    @abstractmethod
    def _compress(
        cls: Type[Self],
        data: Data,
        model: Model | None,
        tokenizer: PreTrainedTokenizer | None,
    ) -> tuple[CompressedData, Metadata]:
        pass

    @classmethod
    @abstractmethod
    def _decompress(
        cls: Type[Self],
        compressed_data: CompressedData,
        model: Model | None,
        tokenizer: PreTrainedTokenizer | None,
    ) -> tuple[Data, Metadata]:
        pass

    @classmethod
    @lru_cache
    def compress(
        cls: Type[Self],
        data: Data,
        model: Model | None,
        tokenizer: PreTrainedTokenizer | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[CompressedData, Metadata]:
        return cls._compress(data, model, tokenizer, *args, **kwargs)

    @classmethod
    @lru_cache
    def decompress(
        cls: Type[Self],
        compressed_data: CompressedData | None,
        model: Model | None,
        tokenizer: PreTrainedTokenizer | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Data, Metadata]:
        return cls._decompress(compressed_data, model, tokenizer, *args, **kwargs)
