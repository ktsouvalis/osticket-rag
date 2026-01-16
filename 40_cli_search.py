from rag_core import RagEngine


def _format_results(results: list[dict]) -> str:
    if not results:
        return "No related tickets found."

    lines = []
    for idx, item in enumerate(results, start=1):
        score = item.get("top_score")
        score_txt = f"{score:.4f}" if isinstance(score, float) else "n/a"
        header = f"{idx}. {item.get('source_type')} #{item.get('ticket_number')} - {item.get('subject') or 'No subject'}  (score: {score_txt})"
        lines.append(header)
        if item.get("url"):
            lines.append(f"   URL: {item['url']}")
        lines.append("")
    return "\n".join(lines).strip()


if __name__ == "__main__":
    engine = RagEngine()
    user_input = input("How can I help you today? ")
    results = engine.retrieve_related(user_input)
    print("\n" + _format_results(results))
