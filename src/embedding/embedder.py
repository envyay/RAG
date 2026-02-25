# from langchain_huggingface import HuggingFaceEmbeddings
#
#
# def get_embedder():
#     return HuggingFaceEmbeddings(
#         # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#         model_name="intfloat/multilingual-e5-large"
#     )

from openai import OpenAI
from typing import List


class LMStudioEmbedder:
    def __init__(self, model_name: str):
        self.client = OpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lm-studio"
        )
        self.model_name = model_name

    # embed query (câu hỏi người dùng)
    def embed_query(self, text: str) -> List[float]:
        text = "query: " + text.strip()

        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )

        return response.data[0].embedding

    # embed documents (passages)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = ["passage: " + t.strip() for t in texts]

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )

        return [d.embedding for d in response.data]