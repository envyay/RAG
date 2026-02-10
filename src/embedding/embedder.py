from langchain_huggingface import HuggingFaceEmbeddings


def get_embedder():
    return HuggingFaceEmbeddings(
        # model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        model_name="intfloat/multilingual-e5-large"
    )
