import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import triton_python_backend_utils as pb_utils

# HF repo of the merged model (set via env on server if needed)
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "Moamineelhilali/deepseek-stackoverflow-merged")
DTYPE = torch.float16

class TritonPythonModel:
    def initialize(self, args):
        token = os.environ.get("HF_TOKEN")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                HF_MODEL_ID, token=token, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL_ID,
                torch_dtype=DTYPE,
                device_map="auto",
                token=token,
                trust_remote_code=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model '{HF_MODEL_ID}'. If this is a private repo, set HF_TOKEN. "
                f"If offline, mount a local snapshot and set HF_MODEL_ID to the mounted path. Original error: {e}"
            )
        self.model.eval()

    def execute(self, requests):
        responses = []
        for request in requests:
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
            # BYTES input: array of python bytes objects
            prompts = [p.decode("utf-8") for p in prompt_tensor.as_numpy().flatten().tolist()]
            outs = []
            for p in prompts:
                inputs = self.tokenizer(
                    p, return_tensors="pt", truncation=True, max_length=512
                ).to(self.model.device)
                with torch.no_grad():
                    gen = self.model.generate(
                        **inputs, max_new_tokens=200, do_sample=True, top_p=0.9, temperature=0.7
                    )
                text = self.tokenizer.decode(gen[0], skip_special_tokens=True)
                outs.append(text.encode("utf-8"))

            out_tensor = pb_utils.Tensor("text", np.array(outs, dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_tensor]))
        return responses
