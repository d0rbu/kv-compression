from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.2-3B",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        # quantization_config=BitsAndBytesConfig(
        #     load_in_8bit=True,
        # ),
    )

    return model, tokenizer
