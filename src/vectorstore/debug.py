# def debug_similarity(vectorstore, query, k=5):
#     docs_scores = vectorstore.similarity_search_with_score(query, k=k)
#
#     print("\n=== RETRIEVER DEBUG ===")
#     for i, (doc, score) in enumerate(docs_scores, 1):
#         print(f"[{i}] score={score:.4f} | {doc.metadata.get('source')}")
#         print(doc.page_content[:200])
#         print("-" * 50)
#
#     return docs_scores
