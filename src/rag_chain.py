"""
rag_chain.py — Retrieval-Augmented Generation (RAG) pipeline.

Architecture:
  User Query
      │
      ▼
  Retriever (ChromaDB similarity search)
      │  top-k relevant chunks
      ▼
  Context Builder (format + deduplicate)
      │
      ▼
  Prompt (system + history + context + question)
      │
      ▼
  LLM (OpenAI / Groq) — streaming
      │
      ▼
  Answer + Source Citations

Features:
  • Streaming responses
  • Conversation memory (last N turns)
  • Source citation extraction
  • Confidence / relevance scoring
  • Follow-up question suggestions
"""
from __future__ import annotations
from typing import List, Dict, Iterator, Optional, Tuple
from datetime import datetime

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import settings
from src.llm import get_llm
from src.vectorstore import get_vector_store
from src.logger import logger


# ── System Prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI Research Assistant. Your job is to answer questions accurately using the provided research documents.

## Guidelines
1. **Ground your answers** in the provided context. If the context doesn't contain enough information, say so clearly.
2. **Cite your sources** — mention the document name or URL when using information from it.
3. **Be comprehensive** — structure longer answers with clear sections when appropriate.
4. **Be honest** — if you're uncertain, say so. Never fabricate citations or facts.
5. **Stay focused** — answer the user's actual question, not a different one.

## Context Format
Documents are provided as [SOURCE: name] followed by the content. Use these sources in your answer.

## Response Format
- Use markdown for formatting (headers, bullets, bold) when it improves clarity.
- For complex topics, structure your answer with sections.
- End with a "**Sources used:**" section listing the documents you referenced.

Today's date: {date}
"""

# ── Prompt with history ────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """## Research Context
{context}

---
## Question
{question}

Please answer based on the research context above."""),
])

# ── Follow-up suggestion prompt ────────────────────────────────────────────────
FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful research assistant. Given a question and answer, suggest 3 brief follow-up questions the user might want to ask next. Return ONLY a JSON array of 3 strings. No explanation."),
    ("human", "Question: {question}\n\nAnswer summary: {answer_snippet}"),
])


# ── Context Builder ────────────────────────────────────────────────────────────
def _build_context(docs: List[Document]) -> str:
    """Format retrieved documents into a clean context string."""
    if not docs:
        return "No relevant documents found in the knowledge base."

    seen_content: set = set()
    blocks = []
    for doc in docs:
        # Deduplicate near-identical chunks
        snippet = doc.page_content[:100]
        if snippet in seen_content:
            continue
        seen_content.add(snippet)

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        page_info = f" (page {page})" if page else ""

        blocks.append(
            f"[SOURCE: {source}{page_info}]\n{doc.page_content.strip()}\n"
        )

    return "\n---\n".join(blocks)


def _extract_sources(docs: List[Document]) -> List[Dict]:
    """Extract unique source metadata from retrieved docs."""
    seen = set()
    sources = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        if src not in seen:
            seen.add(src)
            sources.append({
                "source": src,
                "type": doc.metadata.get("source_type", "unknown"),
                "page": doc.metadata.get("page"),
                "title": doc.metadata.get("title", src),
            })
    return sources


# ── Conversation Memory ────────────────────────────────────────────────────────
class ConversationMemory:
    """Simple sliding-window conversation memory."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._history: List[Dict] = []  # {"role": "human"/"ai", "content": "..."}

    def add_turn(self, human: str, ai: str):
        self._history.append({"role": "human", "content": human})
        self._history.append({"role": "ai", "content": ai})
        # Keep only last N turns
        if len(self._history) > self.max_turns * 2:
            self._history = self._history[-(self.max_turns * 2):]

    def to_langchain_messages(self) -> List:
        messages = []
        for msg in self._history:
            if msg["role"] == "human":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages

    def clear(self):
        self._history = []

    @property
    def turn_count(self) -> int:
        return len(self._history) // 2


# ── RAG Engine ─────────────────────────────────────────────────────────────────
class RAGEngine:
    """Core RAG engine — retrieves, builds context, and generates answers."""

    def __init__(self):
        self.llm = get_llm()
        self.vs = get_vector_store()
        self.memory = ConversationMemory(max_turns=6)
        self._chain = RAG_PROMPT | self.llm | StrOutputParser()
        self._followup_chain = FOLLOWUP_PROMPT | self.llm | StrOutputParser()
        logger.info("RAGEngine initialised")

    def retrieve(self, query: str) -> Tuple[List[Document], List[Dict]]:
        """Retrieve relevant documents and return (docs, source_metadata)."""
        docs_with_scores = self.vs.search_with_scores(query, k=settings.top_k_results)

        # Filter low-relevance results (score < 0.3)
        docs = [doc for doc, score in docs_with_scores if score >= 0.25]
        if not docs:
            docs = [doc for doc, _ in docs_with_scores[:3]]  # fallback: take top 3

        sources = _extract_sources(docs)
        logger.info(f"Retrieved {len(docs)} chunks from {len(sources)} sources")
        return docs, sources

    def stream_answer(
        self,
        question: str,
        source_filter: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Stream the answer token by token.
        Yields string chunks as they arrive from the LLM.
        """
        if self.vs.is_empty():
            yield "⚠️ The knowledge base is empty. Please upload documents or add URLs first."
            return

        # Retrieve relevant docs
        if source_filter:
            docs = self.vs.search(question, source_filter=source_filter)
            sources = _extract_sources(docs)
        else:
            docs, sources = self.retrieve(question)

        context = _build_context(docs)

        # Build input
        chain_input = {
            "context": context,
            "question": question,
            "chat_history": self.memory.to_langchain_messages(),
            "date": datetime.now().strftime("%B %d, %Y"),
        }

        # Stream
        full_answer = ""
        try:
            for chunk in self._chain.stream(chain_input):
                full_answer += chunk
                yield chunk
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            yield f"\n\n⚠️ Error generating response: {e}"
            return

        # Save to memory
        self.memory.add_turn(question, full_answer)
        logger.info(f"Answer generated — {len(full_answer)} chars, memory={self.memory.turn_count} turns")

        # Store sources for UI to display
        self._last_sources = sources
        self._last_docs = docs

    def get_follow_up_questions(self, question: str, answer: str) -> List[str]:
        """Generate 3 follow-up question suggestions."""
        try:
            import json
            result = self._followup_chain.invoke({
                "question": question,
                "answer_snippet": answer[:500],
            })
            # Strip markdown fences if present
            result = result.strip().strip("```json").strip("```").strip()
            questions = json.loads(result)
            if isinstance(questions, list):
                return [str(q) for q in questions[:3]]
        except Exception as e:
            logger.warning(f"Follow-up generation failed: {e}")
        return []

    def clear_memory(self):
        self.memory.clear()
        logger.info("Conversation memory cleared")

    @property
    def last_sources(self) -> List[Dict]:
        return getattr(self, "_last_sources", [])

    @property
    def last_docs(self) -> List[Document]:
        return getattr(self, "_last_docs", [])


# ── Summarisation ──────────────────────────────────────────────────────────────
SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a research summariser. Create a comprehensive, well-structured summary of the provided document content. Use headers and bullet points for clarity."),
    ("human", "Document: {source_name}\n\nContent:\n{content}\n\nPlease provide a comprehensive summary with key points, main themes, and important findings."),
])


def summarise_source(source_name: str) -> Iterator[str]:
    """Stream a summary of all chunks from a specific source."""
    vs = get_vector_store()
    llm = get_llm()

    docs = vs.search("main topics key findings overview", source_filter=source_name, k=8)
    if not docs:
        yield f"No content found for source: {source_name}"
        return

    content = "\n\n---\n\n".join(d.page_content for d in docs[:6])
    chain = SUMMARY_PROMPT | llm | StrOutputParser()

    for chunk in chain.stream({"source_name": source_name, "content": content}):
        yield chunk
