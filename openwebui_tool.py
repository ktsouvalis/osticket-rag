"""
title: osTicket RAG Search
author: ktsouvalis
description: Searches the osTicket knowledge base for relevant tickets and FAQs using semantic search. Returns full ticket threads with source URLs.
required_open_webui_version: 0.4.0
requirements: requests
version: 1.1.0
licence: MIT
"""

import requests
from pydantic import BaseModel, Field


class Tools:
    class Valves(BaseModel):
        rag_api_url: str = Field(
            default="http://195.251.13.132:8800",
            description="Base URL of the osTicket RAG API",
        )
        rag_api_key: str = Field(
            default="",
            description="API key for the RAG API (X-API-Key header)",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.citation = True

    async def search_tickets(self, query: str) -> str:
        """
        Search the osTicket knowledge base for tickets and FAQs related to the query.
        Use this tool when the user asks about past issues, incidents, network problems,
        infrastructure changes, or any topic that may have been documented in a support ticket.
        :param query: The search query describing what to look for.
        :return: Relevant ticket threads with their content and source URLs.
        """
        headers = {}
        if self.valves.rag_api_key:
            headers["X-API-Key"] = self.valves.rag_api_key

        try:
            resp = requests.get(
                f"{self.valves.rag_api_url}/ask",
                params={"query": query},
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            return f"Error contacting RAG API: {e}"

        data = resp.json()
        results = data.get("results", [])

        if not results:
            return "No relevant tickets found for this query."

        output_parts = []
        for i, r in enumerate(results, 1):
            ticket_number = r.get("ticket_number", "?")
            source_type = r.get("source_type", "ticket")
            subject = r.get("subject", "")
            url = r.get("url", "")
            context = r.get("context", "")

            lines = [f"=== Source {i}: {source_type.upper()} #{ticket_number} ==="]
            lines.append(f"Subject: {subject}")
            if url:
                lines.append(f"URL: {url}")
            lines.append("")
            lines.append(context)

            output_parts.append("\n".join(lines))

        return "\n\n---\n\n".join(output_parts)
