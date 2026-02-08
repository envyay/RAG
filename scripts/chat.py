from src.rag.pipeline import rag_answer

DEBUG = True   # bật khi cần xem source

while True:
    q = input("You: ").strip()
    if q.lower() in ["exit", "quit"]:
        break

    result = rag_answer(q)

    print("Bot:", result["answer"])

    if DEBUG and result["sources"]:
        print("\n📚 Sources:")
        for s in result["sources"]:
            print(f"- {s}")

    print()
