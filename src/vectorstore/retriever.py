# from src.embedding.embedder import get_embedder
from src.vectorstore.qdrant_store import QdrantStore

from src.embedding.embedder import LMStudioEmbedder

class Retriever:
    def __init__(self, collection_name="docs"):
        self.embedder = LMStudioEmbedder(
            "text-embedding-intfloat-multilingual-e5-large-instruct"
        )
        ...
        self.qdrant = QdrantStore(
            host="localhost",
            port=6333,
            collection_name=collection_name,
        )

    def retrieve(self, query, top_k=3):
        query_vector = self.embedder.embed_query(query)

        results = self.qdrant.search(query_vector=query_vector, top_k=top_k)

        return results