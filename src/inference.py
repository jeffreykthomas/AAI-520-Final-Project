from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
import torch


class InferenceEngine:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_inference(self, input_text):
        generation_config = GenerationConfig(
            max_new_tokens=250, do_sample=True, top_k=50, eos_token_id=self.model.config.eos_token_id,
            temperature=0.8, pad_token_id=2, num_return_sequences=1, min_new_tokens=30, repetition_penalty=1.2,
        )

        self.model.generation_config = generation_config
        inputs = self.tokenizer(input_text, return_tensors="pt")
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        outputs = self.model.generate(**inputs)

        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # remove the inputs from outputs
        decoded_output = decoded_output.replace(input_text + ' Expert: ', '')

        return decoded_output
