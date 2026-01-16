import os
import re

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ollama import Client


def _strip_html(text: str) -> str:
    if not text or ("<" not in text and ">" not in text):
        return text
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator="\n")


def _is_protected_line(line: str) -> bool:
    if not line:
        return False
    return bool(
        re.search(
            r"\[src:[^\]]+\]|https?://\S+|\bTicket\s+#\d+\b|\bFAQ-\d+\b",
            line,
        )
    )


def _translate_block(client: Client, model: str, text: str) -> str:
    if not text.strip():
        return text
    prompt = (
        "Translate the following text to Greek. Preserve code blocks and URLs, technical terms, "
        "exactly as written. Return only the translated text.\n\n"
        f"TEXT:\n{text}"
    )
    out = client.generate(
        model=model,
        prompt=prompt,
        stream=False,
        options={"temperature": 0},
    )
    return (out.get("response") or "").strip()


def send_to_translation_service(text: str) -> str:
    load_dotenv()
    if not text:
        return ""
    server_ip = os.getenv("SERVER_IP")
    if not server_ip:
        raise RuntimeError("SERVER_IP is not set in environment (.env).")
    translation_model = os.getenv("TRANSLATION_MODEL", "aya-expanse:8b")
    client = Client(host=f"http://{server_ip}:11434")

    clean_text = _strip_html(text)

    lines = clean_text.splitlines()
    out_lines = []
    block_lines = []

    def flush_block():
        if not block_lines:
            return
        block_text = "\n".join(block_lines)
        translated = _translate_block(client, translation_model, block_text)
        out_lines.extend(translated.splitlines())
        block_lines.clear()

    for line in lines:
        if _is_protected_line(line):
            flush_block()
            out_lines.append(line)
        else:
            block_lines.append(line)

    flush_block()
    return "\n".join(out_lines).strip()
