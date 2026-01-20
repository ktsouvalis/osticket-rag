"""Interactive CLI for the OSTicket retriever (retrieval-only).

Run:
    python cli_search.py
"""

from __future__ import annotations

from rag_core import RagEngine


def _format_response(resp: dict) -> str:
    related = resp.get("related") or []
    if not related:
        return "No related tickets found."

    lines: list[str] = []
    for i, doc in enumerate(related, start=1):
        subject = doc.get("subject", "")
        ticket_number = doc.get("ticket_number", "")
        url = doc.get("url", "")
        top_score = doc.get("top_score")
        score_str = f"{top_score:.4f}" if isinstance(top_score, (int, float)) else "n/a"

        lines.append(f"{i}. ticket #{ticket_number} - {subject} (score: {score_str})")
        if url:
            lines.append(f"   URL: {url}")

        evidence = doc.get("evidence") or []
        for ev in evidence[:3]:
            chunk_index = ev.get("chunk_index")
            ev_score = ev.get("score")
            ev_score_str = f"{ev_score:.4f}" if isinstance(ev_score, (int, float)) else "n/a"
            text = (ev.get("text") or "").strip().replace("\r", "")
            preview = text[:280] + ("..." if len(text) > 280 else "")
            lines.append(f"      - chunk {chunk_index} (score: {ev_score_str}): {preview}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    engine = RagEngine()
    user_input = input("How can I help you today? ").strip()
    resp = engine.query(user_input)
    print(_format_response(resp))


if __name__ == "__main__":
    main()
