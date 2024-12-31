from core.model import get_model_and_tokenizer
from arguably import arguably
from enum import Enum


class OptimizationTarget(Enum):
    token_continuous = "token_continuous"
    token_discrete = "token_discrete"
    kv = "kv"

# dynamically build an OptimizationTarget enum based on the files available in core/optimization
# so each file in that directory should be an enum member
# e.g. core/optimization/token_continuous.py




@arguably.command
def main(
    model: str = "meta-llama/Llama-3.2-3B",
    opt_target: OptimizationTarget = OptimizationTarget.kv,
    progressive: bool = False,
):
    assert opt_target == OptimizationTarget.kv, "token optimization is not supported yet"
    assert not progressive, "progressive coding optimization is not supported yet"

    model, tokenizer = get_model_and_tokenizer()
    print(model)
    print(tokenizer)


if __name__ == "__main__":
    arguably(main)
