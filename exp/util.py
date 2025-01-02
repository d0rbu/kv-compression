import os
from enum import Enum
from importlib import import_module

from core.optimization.base import OptimizedRepresentation

CORE = os.path.join(os.path.dirname(__file__), "..", "core")
OPTIMIZATION = os.path.join(CORE, "optimization")


def _get_module_strategy(path: str) -> OptimizedRepresentation | None:
    module = import_module(path)
    module_strategy = None

    for klass in module.__dict__.values():
        if not isinstance(klass, type) or not issubclass(
            klass, OptimizedRepresentation
        ):
            continue

        if module_strategy is not None:
            raise ValueError(f"Multiple optimization strategies in {file}")

        if klass is OptimizedRepresentation:  # skip the base class
            continue

        module_strategy = klass

    return module_strategy


optimization_files = [file for file in os.listdir(OPTIMIZATION) if file.endswith(".py")]
optimization_strategies = {}

for file in optimization_files:
    filename = file[:-3]

    module_strategy = _get_module_strategy(f"core.optimization.{filename}")

    if module_strategy is not None:
        optimization_strategies[filename] = module_strategy

# dynamically add the optimization strategies to the OptimizationStrategy enum
OptimizationStrategy = Enum("OptimizationStrategy", optimization_strategies)

# change optimization strategy keys to be the enum values instead of the filenames
optimization_strategies = {
    OptimizationStrategy[filename]: strategy
    for filename, strategy in optimization_strategies.items()
}
