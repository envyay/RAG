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

# import torch
# from transformers import pipeline
#
# class HFLLM:
#     def __init__(self, model_name):
#         self.pipe = pipeline(
#             "text-generation",
#             model=model_name,
#             tokenizer=model_name,
#             device="cuda",
#             dtype=torch.bfloat16
#         )
#
#     def generate(self, prompt):
#         out = self.pipe(
#             prompt,
#             max_new_tokens=80,
#             temperature=0.2,
#             do_sample=False
#         )
#         return out[0]["generated_text"]

from openai import OpenAI


class LMStudioLLM:
    def __init__(self, model_name: str):
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Bạn là trợ lý AI. Chỉ trả lời dựa trên context được cung cấp.\n\n" + prompt
                }
            ],
            temperature=0.3,
            max_tokens=512
        )

        return response.choices[0].message.content
