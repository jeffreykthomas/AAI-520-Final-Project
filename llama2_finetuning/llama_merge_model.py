from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import os
import torch

output_dir = "../results/"

# Where to load model results
final_checkpoint_dir = os.path.join(output_dir, "final_checkpoint_3epochs")

# Load the entire model on the GPU 0
reloaded_model = AutoPeftModelForCausalLM.from_pretrained(
    final_checkpoint_dir,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map='auto',
)
reloaded_tokenizer = AutoTokenizer.from_pretrained(final_checkpoint_dir)

# Merge the LoRA and the base model
merged_model = reloaded_model.merge_and_unload()