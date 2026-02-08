from src.loaders.text_loader import load_texts
from src.chunking.splitter import split_docs
from src.embedding.embedder import get_embedder
from src.vectorstore.faiss_store import build_index, save_index

docs = load_texts("data/raw")
chunks = split_docs(docs)

embedder = get_embedder()
index = build_index(chunks, embedder)
save_index(index)

print("✅ Ingest xong")
