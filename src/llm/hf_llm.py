from transformers import pipeline
class HFLLM:
    def __init__(self, model_name):
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=300,
            temperature=0.2,
            device="cuda"
        )

    def generate(self, prompt):
        return self.pipe(prompt)[0]["generated_text"]
