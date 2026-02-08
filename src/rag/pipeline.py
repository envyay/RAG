from src.embedding.embedder import get_embedder
from src.llm.hf_llm import HFLLM
from src.rag.context_build import build_context
from src.vectorstore.faiss_store import load_index
from src.vectorstore.retriever import get_retriever

_embedder = get_embedder()
_vectorstore = load_index(_embedder)
_retriever = get_retriever(_vectorstore)
_llm = HFLLM("google/gemma-3-4b-it")

def rag_answer(question: str):
    docs = _retriever.invoke(question)
    if not docs:
        return {
            "answer": "Tôi không tìm thấy thông tin trong tài liệu.",
            "sources": [],
            "query": question
        }

    context, used_docs = build_context(docs, question)

    if not context.strip() or len(context.strip()) < 50:
        return {
            "answer": "Tôi không tìm thấy thông tin trong tài liệu.",
            "sources": [],
            "query": question
        }

    with open("src/prompts/rag_prompt.txt", "r", encoding="utf-8") as f:
        prompt = f.read().format(
            question=question,
            context=context
        )

    answer = _llm.generate(prompt)

    if "### ANSWER:" in answer:
        answer = answer.split("### ANSWER:")[-1]

    answer = answer.strip()

    sources = sorted(set(
        d.metadata.get("source", "unknown")
        for d in used_docs
    ))

    return {
        "answer": answer,
        "sources": sources,
        "query": question
    }
