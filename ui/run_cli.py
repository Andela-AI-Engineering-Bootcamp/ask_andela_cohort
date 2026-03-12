from scripts.qa import ask

print("🎓 Andela Cohort QA System\n")

while True:
    question = input("Ask a question (or 'exit'): ")
    if question.lower() in ["exit", "quit"]:
        break

    result = ask(question)

    print("\nAnswer:\n", result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata.get("source", "unknown"))