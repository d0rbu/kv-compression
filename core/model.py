from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig


def get_model_and_tokenizer(
    model_name: str = "meta-llama/Llama-3.2-3B",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_8bit=True,
        ),
    )

    lora_config = LoraConfig(
        target_modules=["q_proj"],
        init_lora_weights=False,
        r=1,
    )

    model.add_adapter(lora_config, adapter_name="placeholder")
    model.disable_adapters()

    return model, tokenizer
