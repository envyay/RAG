# def build_context(docs):
#     used_docs = docs
#     context = "\n\n".join(d.page_content for d in used_docs)
#     return context, used_docs

def build_context(docs, question, min_overlap=1):
    used_docs = []

    q_tokens = set(question.lower().split())

    for d in docs:
        doc_tokens = set(d.page_content.lower().split())
        overlap = q_tokens & doc_tokens

        if len(overlap) >= min_overlap:
            used_docs.append(d)

    context = "\n\n".join(d.page_content for d in used_docs)
    return context, used_docs
