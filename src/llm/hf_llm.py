# import torch
# from transformers import pipeline
# class HFLLM:
#     def __init__(self, model_name):
#         self.pipe = pipeline(
#             "text-generation",
#             model=model_name,
#             tokenizer=model_name,
#             max_new_tokens=80,
#             temperature=0.2,
#             device="cuda",
#             dtype=torch.bfloat16
#         )
#
#     def generate(self, prompt):
#         return self.pipe(prompt)[0]["generated_text"]

import torch
from transformers import pipeline

class HFLLM:
    def __init__(self, model_name):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            device="cuda",
            dtype=torch.bfloat16
        )

    def generate(self, prompt):
        out = self.pipe(
            prompt,
            max_new_tokens=80,
            temperature=0.2,
            do_sample=False
        )
        return out[0]["generated_text"]
