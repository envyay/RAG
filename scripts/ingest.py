from src.loaders.text_loader import load_texts
from src.chunking.splitter import split_docs
from src.embedding.embedder import get_embedder
from src.vectorstore.qdrant_store import QdrantStore

def main():
    docs = load_texts("data/raw")
    chunks = split_docs(docs)

    embedder = get_embedder()
    embeddings = embedder.embed_documents(c.page_content for c in chunks)

    qdrant = QdrantStore(
        host="localhost",
        port=6333,
        collection_name="docs",
    )

    qdrant.create_collection(vector_size=len(embeddings[0]))
    qdrant.add_doc(embeddings, chunks)

    print("✅ Ingest xong")

if __name__ == "__main__":
    main()