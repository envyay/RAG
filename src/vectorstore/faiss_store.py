from langchain_community.vectorstores import FAISS


def build_index(docs, embedder):
    return FAISS.from_documents(docs, embedder)

def save_index(index, path="embeddings/faiss"):
    index.save_local(path)

def load_index(embedder, path="embeddings/faiss"):
    return FAISS.load_local(
        path,
        embedder,
        allow_dangerous_deserialization=True
    )
