import os
import re
from collections import defaultdict
from typing import Optional

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from ollama import Client
from pymilvus import Collection, connections


# Heuristic to detect “broad / whole process / list all” questions
BROAD_QUERY_RE = re.compile(
    r"\b(which|list|all|enumerate|everything|whole|overall|during|process|timeline|summary|όλα|ολα|ποιες|ποια|λίστα|λιστα)\b",
    re.IGNORECASE,
)

# Ticket number sanitization
_TICKETNO_RE = re.compile(r"(\d{3,})")

# Citation extraction from model output:
# e.g. [src: ticket #000225 chunk:2 pk:123 score:0.8123]
CITATION_RE = re.compile(r"\[src:\s*([A-Za-z0-9_-]+)\s+#([^\s\]]+)", re.IGNORECASE)

# VLAN enumeration mode
ENUM_QUERY_RE = re.compile(
    r"\b(which|list|all|created|new|during|redesign|process|ανασχεδιασμ|ποιες|ποια|ολα|όλα|λιστα|λίστα)\b",
    re.IGNORECASE,
)
VLAN_PATTERNS = [
    re.compile(r"(?i)\bvlan\s*[:#]?\s*([0-9]{2,5})\b"),
    re.compile(r"(?i)\bvlanif\s*([0-9]{2,5})\b"),
    re.compile(r"(?i)\bvlan([0-9]{2,5})\b"),
]


def _ensure_vector_batch(embeddings):
    # Ollama may return either [vector] or [[vector]] depending on input
    if embeddings and isinstance(embeddings, list) and isinstance(embeddings[0], float):
        return [embeddings]
    return embeddings


def _escape_milvus_str(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def sanitize_ticket_number(ticket_number, ticket_id):
    """
    Ensure ticket_number is printable/stable.
    Prefer digits; fallback to ticket_id; finally 'unknown'.
    """
    if ticket_number is None:
        return str(ticket_id) if ticket_id is not None else "unknown"

    s = str(ticket_number).strip()
    m = _TICKETNO_RE.search(s)
    if m:
        digits = m.group(1)
        if len(digits) <= 6:
            return digits.zfill(6)
        return digits

    return str(ticket_id) if ticket_id is not None else "unknown"


def extract_used_doc_keys_from_answer(answer_text: str) -> set[str]:
    used = set()
    if not answer_text:
        return used
    for m in CITATION_RE.finditer(answer_text):
        source_type = (m.group(1) or "").strip().lower()
        ticket_number = (m.group(2) or "").strip()
        used.add(f"{source_type}:{ticket_number}")
    return used


def is_broad_query(q: str) -> bool:
    q = (q or "").strip()
    return bool(q) and bool(BROAD_QUERY_RE.search(q))


def is_vlan_enumeration_query(q: str) -> bool:
    q = (q or "").strip()
    return bool(q) and ("vlan" in q.lower()) and bool(ENUM_QUERY_RE.search(q))


def extract_vlan_ids(text: str) -> set[str]:
    if not text:
        return set()
    out = set()
    for pat in VLAN_PATTERNS:
        for m in pat.finditer(text):
            out.add(m.group(1))
    return out


def redact_secrets(text: str) -> str:
    if not text:
        return ""
    patterns = [
        r"(\b(?:admin|root|netadmin|user)\b)\s*/\s*([^\s\)]+)",
        r"(?i)(password\s*:\s*)(\S+)",
        r"(?i)(only\s+password\s*:\s*)(\S+)",
    ]
    redacted = text
    redacted = re.sub(patterns[0], r"\1 / [REDACTED]", redacted)
    redacted = re.sub(patterns[1], r"\1[REDACTED]", redacted)
    redacted = re.sub(patterns[2], r"\1[REDACTED]", redacted)
    return redacted


def clean_html_for_llm(html_content: str) -> str:
    # Kept for completeness; not used in this module
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup.find_all(["p", "br", "div", "li", "tr", "h1", "h2", "h3"]):
        tag.append("\n")
    text = soup.get_text(separator=" ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text.strip()


def pick_chunk_indices_for_doc(items, *, top_unique: int, neighbor_window: int, max_total: int):
    wanted = []
    seen_idx = set()

    for score, pk, ticket_id, ticket_number, source_type, chunk_index, subject, payload in sorted(
        items, key=lambda x: x[0], reverse=True
    ):
        if not isinstance(chunk_index, int):
            continue
        if chunk_index in seen_idx:
            continue
        wanted.append(chunk_index)
        seen_idx.add(chunk_index)
        if len(wanted) >= top_unique:
            break

    expanded = set([0])
    for i in wanted:
        for d in range(-neighbor_window, neighbor_window + 1):
            j = i + d
            if j >= 0:
                expanded.add(j)

    idxs = sorted(expanded)
    return idxs[:max_total]


class RagEngine:
    def __init__(
        self,
        *,
        server_ip: Optional[str] = None,
        collection_name: str = "osticket_knowledge",
    ):
        load_dotenv()

        self.server_ip = server_ip or os.getenv("SERVER_IP")
        if not self.server_ip:
            raise RuntimeError("SERVER_IP is not set in environment (.env).")

        self.base_ticket_url = (os.getenv("BASE_TICKET_URL", "") or "").strip()
        self.embed_model = os.getenv("EMBED_MODEL_NAME", "bge-m3")
        self.response_model = os.getenv("RESPONSE_MODEL_NAME", "qwen2.5:14b")

        # Retrieval knobs
        self.search_limit = int(os.getenv("RAG_SEARCH_LIMIT", "120"))
        self.max_docs = int(os.getenv("RAG_MAX_DOCS", "8"))
        self.top_chunks_per_doc = int(os.getenv("RAG_TOP_CHUNKS_PER_DOC", "4"))
        self.neighbor_window = int(os.getenv("RAG_NEIGHBOR_WINDOW", "1"))
        self.max_context_chars = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "24000"))
        self.nprobe = int(os.getenv("RAG_NPROBE", "10"))

        # Broad-query adaptation
        self.broad_query_boost = float(os.getenv("RAG_BROAD_QUERY_BOOST", "2.5"))
        self.broad_top_unique_mult = int(os.getenv("RAG_BROAD_TOP_UNIQUE_MULT", "3"))
        self.broad_neighbor_min = int(os.getenv("RAG_BROAD_NEIGHBOR_MIN", "2"))
        self.max_indices_per_doc = int(os.getenv("RAG_MAX_INDICES_PER_DOC", "80"))

        # VLAN mode
        self.vlan_enum_limit_per_query = int(os.getenv("RAG_VLAN_LIMIT_PER_QUERY", "300"))
        self.vlan_enum_max_results = int(os.getenv("RAG_VLAN_MAX_RESULTS", "3000"))

        # Connections
        self.ollama = Client(host=f"http://{self.server_ip}:11434")
        connections.connect(host=self.server_ip, port="19530")
        self.collection = Collection(collection_name)
        self.collection.load()

    def fetch_chunks_by_indices(self, ticket_id: int, source_type: str, indices: list[int]):
        if ticket_id is None or not indices:
            return []

        idxs = sorted({int(i) for i in indices if isinstance(i, int) or (isinstance(i, str) and i.isdigit())})
        idxs = [i for i in idxs if i >= 0]
        if not idxs:
            return []

        idxs_csv = ", ".join(str(i) for i in idxs)
        st = _escape_milvus_str(source_type or "ticket")
        expr = f"ticket_id == {int(ticket_id)} && source_type == '{st}' && chunk_index in [{idxs_csv}]"

        rows = self.collection.query(
            expr=expr,
            output_fields=[
                "pk",
                "ticket_id",
                "ticket_number",
                "source_type",
                "chunk_index",
                "subject",
                "text_payload",
            ],
        )

        out = []
        for r in rows:
            payload = (r.get("text_payload") or "").strip()
            if not payload:
                continue
            out.append(
                (
                    None,  # score unknown for query() results
                    int(r["pk"]),
                    r.get("ticket_id"),
                    sanitize_ticket_number(r.get("ticket_number"), r.get("ticket_id")),
                    r.get("source_type") or "ticket",
                    r.get("chunk_index"),
                    r.get("subject") or "",
                    payload,
                )
            )
        return out

    def milvus_multi_search(self, queries: list[str], limit_per_query: int):
        best_by_pk = {}
        for q in queries:
            q_vec = self.ollama.embed(model=self.embed_model, input=q)["embeddings"]
            q_vec = _ensure_vector_batch(q_vec)

            results = self.collection.search(
                data=q_vec,
                anns_field="vector",
                param={"metric_type": "COSINE", "params": {"nprobe": self.nprobe}},
                limit=limit_per_query,
                output_fields=[
                    "pk",
                    "ticket_id",
                    "ticket_number",
                    "source_type",
                    "chunk_index",
                    "subject",
                    "text_payload",
                ],
            )

            for hits in results:
                for hit in hits:
                    ent = hit.entity
                    payload = (ent.get("text_payload") or "").strip()
                    if not payload:
                        continue

                    pk = int(ent.get("pk"))
                    score = float(hit.distance)

                    ticket_id = ent.get("ticket_id")
                    source_type = ent.get("source_type") or "ticket"
                    chunk_index = ent.get("chunk_index")
                    subject = ent.get("subject") or ""
                    ticket_number = sanitize_ticket_number(ent.get("ticket_number"), ticket_id)

                    row = (score, pk, ticket_id, ticket_number, source_type, chunk_index, subject, payload)
                    if pk not in best_by_pk or score > best_by_pk[pk][0]:
                        best_by_pk[pk] = row

        return list(best_by_pk.values())

    def answer_vlan_enumeration(self, user_query: str) -> str:
        queries = [
            user_query,
            "network redesign vlan",
            "ανασχεδιασμός δικτύου vlan",
            "δημιουργήθηκαν vlan",
            "new vlans",
            "VLANif",
            "trunk vlan",
        ]

        hits = self.milvus_multi_search(queries, limit_per_query=self.vlan_enum_limit_per_query)
        hits.sort(key=lambda x: x[0], reverse=True)
        hits = hits[: self.vlan_enum_max_results]

        evidence = defaultdict(list)
        for score, pk, ticket_id, ticket_number, source_type, chunk_index, subject, payload in hits:
            vids = extract_vlan_ids(payload)
            if not vids:
                continue
            for vid in vids:
                evidence[vid].append((score, pk, ticket_id, ticket_number, source_type, chunk_index))

        if not evidence:
            return "Sources used: (none)\nNo VLAN IDs found in retrieved chunks."

        rows = []
        for vid, ev in evidence.items():
            ev.sort(key=lambda t: t[0], reverse=True)
            score, pk, ticket_id, ticket_number, source_type, chunk_index = ev[0]
            rows.append((int(vid), score, pk, ticket_id, ticket_number, source_type, chunk_index))
        rows.sort(key=lambda r: r[0])

        used_sources = []
        used_ticket_ids = set()
        top_src = sorted(rows, key=lambda r: r[1], reverse=True)[:12]
        for vlan_id, score, pk, ticket_id, ticket_number, source_type, chunk_index in top_src:
            used_sources.append(f"[src: {source_type} #{ticket_number} chunk:{chunk_index} pk:{pk} score:{score:.4f}]")

        lines = []
        lines.append("Sources used: " + (", ".join(used_sources) if used_sources else "(none)"))
        lines.append("")
        lines.append("VLANs found (extracted):")
        for vlan_id, score, pk, ticket_id, ticket_number, source_type, chunk_index in rows:
            lines.append(f"- VLAN {vlan_id}  [src: {source_type} #{ticket_number} chunk:{chunk_index} pk:{pk} score:{score:.4f}]")
            if self.base_ticket_url and source_type == "ticket" and isinstance(ticket_id, int) and ticket_id < 100000:
                used_ticket_ids.add((ticket_id, ticket_number))

        if self.base_ticket_url and used_ticket_ids:
            lines.append("")
            lines.append("REFERENCE LINKS:")
            for tid, tnum in sorted(used_ticket_ids, key=lambda x: x[1]):
                lines.append(f"Ticket #{tnum}: {self.base_ticket_url}{tid}")

        return "\n".join(lines)

    def answer(self, user_query: str) -> str:
        user_query = (user_query or "").strip()
        if not user_query:
            return "Empty query."

        if is_vlan_enumeration_query(user_query):
            return self.answer_vlan_enumeration(user_query)

        broad = is_broad_query(user_query)

        search_limit = int(self.search_limit * self.broad_query_boost) if broad else self.search_limit
        max_docs = int(self.max_docs * self.broad_query_boost) if broad else self.max_docs
        max_docs = min(max_docs, 25)

        top_unique = (self.top_chunks_per_doc * self.broad_top_unique_mult) if broad else self.top_chunks_per_doc
        neighbor_window = max(self.neighbor_window, self.broad_neighbor_min) if broad else self.neighbor_window

        q_vec = self.ollama.embed(model=self.embed_model, input=user_query)["embeddings"]
        q_vec = _ensure_vector_batch(q_vec)

        search_results = self.collection.search(
            data=q_vec,
            anns_field="vector",
            param={"metric_type": "COSINE", "params": {"nprobe": self.nprobe}},
            limit=search_limit,
            output_fields=[
                "pk",
                "ticket_id",
                "ticket_number",
                "source_type",
                "chunk_index",
                "subject",
                "text_payload",
            ],
        )

        hits_by_doc = defaultdict(list)
        for hits in search_results:
            for hit in hits:
                ent = hit.entity
                payload = (ent.get("text_payload") or "").strip()
                if not payload:
                    continue

                pk = int(ent.get("pk"))
                score = float(hit.distance)

                ticket_id = ent.get("ticket_id")
                ticket_number = sanitize_ticket_number(ent.get("ticket_number"), ticket_id)
                source_type = ent.get("source_type") or "ticket"
                chunk_index = ent.get("chunk_index")
                subject = ent.get("subject") or ""

                doc_key = f"{source_type.lower()}:{ticket_number}"
                hits_by_doc[doc_key].append(
                    (score, pk, ticket_id, ticket_number, source_type, chunk_index, subject, payload)
                )

        ranked_docs = sorted(
            hits_by_doc.items(),
            key=lambda kv: max(item[0] for item in kv[1]),
            reverse=True,
        )

        context_blocks = []
        total_chars = 0

        context_doc_keys = set()
        doc_key_to_label = {}
        doc_key_to_link = {}

        for doc_key, all_items_for_doc in ranked_docs[:max_docs]:
            if not all_items_for_doc:
                continue

            best = max(all_items_for_doc, key=lambda x: x[0])
            _score, _pk, ticket_id, ticket_number, source_type, _chunk_index, subject, _payload = best

            context_doc_keys.add(doc_key)
            doc_key_to_label[doc_key] = f"{source_type} #{ticket_number}"
            if self.base_ticket_url and source_type == "ticket" and isinstance(ticket_id, int) and ticket_id < 100000:
                doc_key_to_link[doc_key] = f"Ticket #{ticket_number}: {self.base_ticket_url}{ticket_id}"

            wanted_idxs = pick_chunk_indices_for_doc(
                all_items_for_doc,
                top_unique=top_unique,
                neighbor_window=neighbor_window,
                max_total=self.max_indices_per_doc,
            )

            fetched = self.fetch_chunks_by_indices(
                ticket_id=int(ticket_id) if ticket_id is not None else None,
                source_type=source_type,
                indices=wanted_idxs,
            )

            score_by_pk = {pk: score for (score, pk, *_rest) in all_items_for_doc}
            fetched.sort(key=lambda x: (x[5] if isinstance(x[5], int) else 10**9))

            chunk_texts = []
            for _s, pk, ticket_id, ticket_number, source_type, chunk_index, subject, payload in fetched:
                score = score_by_pk.get(pk)
                score_txt = f"{score:.4f}" if isinstance(score, float) else "n/a"
                citation = f"[src: {source_type} #{ticket_number} chunk:{chunk_index} pk:{pk} score:{score_txt}]"
                block = f"{citation}\n{redact_secrets(payload)}"

                if total_chars + len(block) > self.max_context_chars:
                    break
                chunk_texts.append(block)
                total_chars += len(block)

            if not chunk_texts:
                continue

            context_blocks.append(
                f"<document source_type='{source_type}' ticket_number='{ticket_number}' ticket_id='{ticket_id}' subject='{subject}'>\n"
                + "\n\n".join(chunk_texts)
                + "\n</document>"
            )

            if total_chars >= self.max_context_chars:
                break

        full_context = "\n\n".join(context_blocks)

        system_prompt = f"""
ROLE: Senior IT/DevOps Engineer at the University of Peloponnese.

STRICT RULES:
1) Use ONLY <context>. Do not add generic steps or assumptions.
2) If the user request conflicts with the context, state the mismatch explicitly.
3) For commands/configuration/code blocks: copy verbatim from the context (do not rewrite).
4) Every bullet/list item must include at least one citation tag like:
   [src: ticket #000225 chunk:2 pk:123 score:0.8123]
5) If the context is insufficient, say what is missing.
6) Most times you will be asked in Greek. ALWAYS RESPOND IN ENGLISH.

OUTPUT:
- Start with: "Sources used: ..." (list the citation tags you actually relied on)
- Then answer.

<context>
{full_context}
</context>

USER QUERY: {user_query}
""".strip()

        out = self.ollama.generate(
            model=self.response_model,
            prompt=system_prompt,
            stream=False,
            options={"temperature": 0, "num_ctx": 32768, "num_predict": 2048},
        )

        full_answer = (out.get("response") or "").strip()

        # Add "used vs not used" references at the end
        used_doc_keys = extract_used_doc_keys_from_answer(full_answer)
        used = sorted([k for k in context_doc_keys if k in used_doc_keys])
        not_used = sorted([k for k in context_doc_keys if k not in used_doc_keys])

        ref_lines = []
        if used:
            ref_lines.append("")
            ref_lines.append("USED REFERENCES (cited in answer):")
            for k in used:
                ref_lines.append(doc_key_to_link.get(k, doc_key_to_label.get(k, k)))

        if not_used:
            ref_lines.append("")
            ref_lines.append("RETRIEVED CONTEXT (not cited):")
            for k in not_used:
                ref_lines.append(doc_key_to_link.get(k, doc_key_to_label.get(k, k)))

        if ref_lines:
            full_answer = full_answer + "\n" + "\n".join(ref_lines)

        return full_answer