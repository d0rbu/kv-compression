from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_and_tokenizer(model_name: str = "meta-llama/Llama-3.2-3B") -> tuple[AutoModelForCausalLM, AutoTokenizer]: 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer
