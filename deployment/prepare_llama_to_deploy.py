from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch

# Procedure based upon
# https://towardsai.net/p/machine-learning/gptq-quantization-on-a-llama-2-7b-fine-tuned-model-with-huggingface

# Set the model to load
hf_model_repo = 'jeffreykthomas/llama2-7b-ubuntu-generation'
# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(hf_model_repo, use_fast=True)
# Set quantization configuration
quantization_config = GPTQConfig(
    bits=4,
    group_size=128,
    dataset="c4",
    desc_act=False,
    tokenizer=tokenizer
)
# Load the model from HF
quant_model = AutoModelForCausalLM.from_pretrained(hf_model_repo,
                                                   quantization_config=quantization_config,
                                                   device_map='auto')

quant_model.push_to_hub("jeffreykthomas/llama2-7b-ubuntu-GPTQ")
tokenizer.push_to_hub("jeffreykthomas/llama2-7b-ubuntu-GPTQ")
