from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, pipeline
import torch
from typing import Any, Dict

dtype = torch.float16


class EndpointHandler:
    def __init__(self, path=""):
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=dtype)

        self.pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=dtype)

    def __call__(self, data: Dict[str, Any]) -> [str]:
        inputs = data.pop("inputs", data)
        generation_config = GenerationConfig(
            max_length=1024,
            max_new_tokens=250, do_sample=True, top_k=50,
            temperature=0.8, pad_token_id=2, num_return_sequences=1,
            min_new_tokens=30, repetition_penalty=1.2,
        )

        output = self.pipeline(inputs, **generation_config.to_dict())

        return output
