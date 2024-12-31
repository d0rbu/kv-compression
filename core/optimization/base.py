from torch import Tensor
from loguru import logger
from abc import ABC, abstractmethod
from typing import Self, Sequence, Mapping, Callable


DataFormat = Sequence[Tensor] | Mapping[str, Tensor] | Tensor


class OptimizedRepresentation(ABC):
    @abstractmethod
    def __init__(
        self: Self,
        data: DataFormat | None = None,
    ) -> None:
        pass

    @property
    @abstractmethod
    def data() -> DataFormat:
        pass

    @abstractmethod
    def data_hooked_generate(
        self: Self,
        **kwargs,
    ) -> Callable:  # curried generate function with data hooked in
        pass
