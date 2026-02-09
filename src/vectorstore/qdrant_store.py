from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid


class QdrantStore:
    def __init__(self, host="localhost", port=6333, collection_name="docs"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def create_collection(self, vector_size: int):
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name not in existing:
            collection = self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

    def add_doc(self, embeddings, documents):
        points = []
        for embedding, document in zip(embeddings, documents):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": document.page_content,
                        "metadata": document.metadata,
                    }
                )
            )
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def search(self, query_vector, top_k=5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        docs = []
        for result in results:
            payload = result.payload
            docs.append(
                Document(
                    page_content=payload.get("text"),
                    metadata=payload.get("metadata"),
                )
            )

            return docs

        def remove_collection(self, collection_name):
            collections = self.client.delete_collection(collection_name)
