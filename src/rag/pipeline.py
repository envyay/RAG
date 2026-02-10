from src.embedding.embedder import get_embedder
from src.llm.hf_llm import HFLLM
from src.rag.context_build import build_context
from src.vectorstore.retriever import Retriever

# retriever = Retriever("docs")

# _embedder = get_embedder()
_retriever = Retriever("docs")
_llm = HFLLM("google/gemma-3-4b-it")

with open("src/prompts/rag_prompt.txt", "r", encoding="utf-8") as f:
    RAG_PROMPT_TEMPLATE = f.read()

def rag_answer(question: str):
    docs = _retriever.retrieve(question)
    if not docs:
        return {
            "answer": "Tôi không tìm thấy thông tin trong tài liệu.",
            "sources": [],
            "query": question
        }

    # context, used_docs = build_context(docs, question)
    context, used_docs = build_context(docs, max_docs=3, max_chars=2500)

    if not context.strip() or len(context.strip()) < 50:
        return {
            "answer": "Tôi không tìm thấy thông tin trong tài liệu.",
            "sources": [],
            "query": question
        }

    prompt = RAG_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
    print("PROMPT LENGTH:", len(prompt))
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
