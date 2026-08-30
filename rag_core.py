
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_REPO_ID = "meta-llama/Llama-3.1-8B-Instruct"
HUGGINGFACE_PROVIDER = "novita"

SUPPORTED_COMPANIES = ["BMW", "Tesla", "Ford"]

FALLBACK_ANSWER = (
    "I could not determine this from the available annual reports."
)

FINANCIAL_SYSTEM_PROMPT = """
You are an AI assistant helping a financial analyst analyse annual reports
from BMW, Tesla and Ford.

Use ONLY the supplied annual-report context.

Rules:
1. Never invent financial figures.
2. Distinguish clearly between revenue, EBIT, EBITDA, profit before tax,
   operating income and net income.
3. Preserve currencies and units.
4. Do not silently convert currencies.
5. If asked for company revenue, do not answer with a regional or segment
   revenue figure unless explicitly requested.
6. Comparative tables in later annual reports can contain valid figures for
   earlier years.
7. For comparison questions, give one clearly labeled value per company.
8. If a company's value cannot be supported by the retrieved context, say so
   for that company only; do not discard values for the other companies.
9. Do not use outside knowledge.
10. Do not mention internal labels such as "Document 1".
11. Be concise and factual.
""".strip()

FOLLOW_UP_SYSTEM_PROMPT = """
Rewrite the newest user question as a standalone retrieval query.

Resolve references such as:
- "compare it with Tesla and BMW"
- "compare these three in 2022"
- "what about Ford?"
- "and in 2022?"
- "what about the previous year?"

If the user refers to "these three", infer BMW, Tesla and Ford when those are
the companies discussed in the conversation.

Do NOT answer.
Return only the rewritten standalone query.
""".strip()


def detect_companies(question: str) -> list[str]:
    q = question.lower()
    companies = []

    if "bmw" in q:
        companies.append("BMW")
    if "tesla" in q:
        companies.append("Tesla")
    if "ford" in q:
        companies.append("Ford")

    if "these three" in q or "all three" in q:
        return SUPPORTED_COMPANIES.copy()

    return companies


def detect_company(question: str) -> str | None:
    companies = detect_companies(question)
    return companies[0] if len(companies) == 1 else None


def detect_year(question: str) -> int | None:
    matches = re.findall(r"\b((?:19|20)\d{2})\b", question)
    return int(matches[0]) if len(matches) == 1 else None


def detect_metric(question: str) -> str | None:
    q = question.lower()

    if (
        "profit before tax" in q
        or "profit/loss before tax" in q
        or "profit / loss before tax" in q
        or "pre-tax profit" in q
    ):
        return "profit before tax"

    if "ebitda" in q:
        return "ebitda"
    if "ebit" in q:
        return "ebit"
    if "net income" in q or "net profit" in q:
        return "net income"
    if "revenue" in q or "revenues" in q or "sales" in q:
        return "revenue"

    return None


@lru_cache(maxsize=1)
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    return FAISS.load_local(
        DB_FAISS_PATH,
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def load_llm():
    token = os.getenv("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN was not found. Add it to your .env file.")

    return InferenceClient(
        provider=HUGGINGFACE_PROVIDER,
        api_key=token,
    )


def call_llm(client, messages, max_tokens=700, temperature=0.1):
    response = client.chat_completion(
        model=HUGGINGFACE_REPO_ID,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    content = response.choices[0].message.content
    return str(content).strip() if content else ""


def format_history(messages, max_messages=8):
    recent = messages[-max_messages:]
    return "\n".join(
        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        for m in recent
    )


def rewrite_question(client, question, messages):
    if len(messages) <= 1:
        return question

    history = format_history(messages[:-1])

    prompt = (
        f"Conversation history:\n{history}\n\n"
        f"Current question:\n{question}\n\n"
        "Standalone query:"
    )

    try:
        rewritten = call_llm(
            client,
            [
                {"role": "system", "content": FOLLOW_UP_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=120,
            temperature=0.01,
        )

        if rewritten:
            return rewritten.strip().strip('"')
    except Exception:
        pass

    return question


def _doc_filename(doc):
    metadata = getattr(doc, "metadata", {}) or {}
    filename = metadata.get("filename")
    if filename:
        return str(filename)

    source = str(metadata.get("source", ""))
    return Path(source).name if source else ""


def _doc_report_year(doc):
    metadata = getattr(doc, "metadata", {}) or {}
    try:
        return int(metadata.get("year"))
    except (TypeError, ValueError):
        return None


def _doc_key(doc):
    metadata = getattr(doc, "metadata", {}) or {}
    page = metadata.get("page_number", metadata.get("page", ""))
    content = getattr(doc, "page_content", "") or ""
    return (_doc_filename(doc), str(page), content[:180])


def _company_matches(doc, company):
    metadata = getattr(doc, "metadata", {}) or {}
    metadata_company = str(metadata.get("company", "")).lower()

    if metadata_company:
        return metadata_company == company.lower()

    return company.lower() in _doc_filename(doc).lower()


def _score_doc(doc, question, company):
    text = (getattr(doc, "page_content", "") or "").lower()
    year = detect_year(question)
    metric = detect_metric(question)
    report_year = _doc_report_year(doc)
    score = 0.0

    if year and re.search(rf"\b{year}\b", text):
        score += 16
    if year and report_year == year:
        score += 6

    if company.lower() in text:
        score += 1

    if metric == "revenue":
        if "company key metrics" in text:
            score += 15
        if "total revenues" in text:
            score += 15
        if "revenue ($m)" in text:
            score += 13
        if "revenue" in text:
            score += 5

        for marker in [
            "europe",
            "north america",
            "south america",
            "china",
            "mobility segment",
            "ford credit segment",
            "automotive segment",
        ]:
            if marker in text and marker not in question.lower():
                score -= 9

    elif metric == "profit before tax":
        if "profit before tax" in text or "profit / loss before tax" in text:
            score += 18
        if "key performance indicators" in text:
            score += 10
        if "bmw group in figures" in text:
            score += 12

    return score


def _get_all_docs(vectorstore):
    docstore = getattr(vectorstore, "docstore", None)
    mapping = getattr(docstore, "_dict", None) if docstore else None
    return list(mapping.values()) if isinstance(mapping, dict) else []


def build_single_company_queries(question, company):
    year = detect_year(question)
    metric = detect_metric(question)

    queries = [question]

    if year and metric == "revenue":
        queries.extend([
            f"{company} {year} company key metrics revenue",
            f"{company} {year} total revenues",
            f"{company} {year} consolidated revenue",
        ])

    if year and metric == "profit before tax":
        queries.extend([
            f"{company} {year} profit before tax",
            f"{company} {year} group profit loss before tax",
        ])

    return list(dict.fromkeys(queries))


def retrieve_documents_for_company(vectorstore, question, company, k=6):
    seen = set()
    candidates = []
    candidate_k = max(30, k * 5)

    def add_docs(docs):
        for doc in docs:
            if not _company_matches(doc, company):
                continue

            key = _doc_key(doc)
            if key not in seen:
                seen.add(key)
                candidates.append(doc)

    # Dense retrieval
    for query in build_single_company_queries(question, company):
        try:
            docs = vectorstore.similarity_search(
                query,
                k=candidate_k,
                filter={"company": company},
            )
        except Exception:
            docs = vectorstore.similarity_search(query, k=candidate_k)

        add_docs(docs)

    # Lexical retrieval
    year = detect_year(question)
    metric = detect_metric(question)

    lexical = []

    for doc in _get_all_docs(vectorstore):
        if not _company_matches(doc, company):
            continue

        text = (getattr(doc, "page_content", "") or "").lower()
        report_year = _doc_report_year(doc)

        if year:
            year_relevant = (
                re.search(rf"\b{year}\b", text)
                or report_year == year
            )
            if not year_relevant:
                continue

        if metric == "revenue" and "revenue" not in text:
            continue

        if metric == "profit before tax" and not (
            "profit" in text and "tax" in text
        ):
            continue

        score = _score_doc(doc, question, company)
        if score > 0:
            lexical.append((score, doc))

    lexical.sort(key=lambda x: x[0], reverse=True)
    add_docs([doc for _, doc in lexical[:candidate_k]])

    ranked = sorted(
        candidates,
        key=lambda d: _score_doc(d, question, company),
        reverse=True,
    )

    return ranked[:k]


def retrieve_documents(vectorstore, question, k=6):
    companies = detect_companies(question)

    if len(companies) == 1:
        return retrieve_documents_for_company(
            vectorstore,
            question,
            companies[0],
            k=k,
        )

    # For multi-company query, retrieve independently per company.
    if len(companies) > 1:
        combined = []
        seen = set()

        per_company_k = max(3, k // len(companies))

        for company in companies:
            docs = retrieve_documents_for_company(
                vectorstore,
                question,
                company,
                k=per_company_k,
            )

            for doc in docs:
                key = _doc_key(doc)
                if key not in seen:
                    seen.add(key)
                    combined.append(doc)

        return combined

    # No company explicitly found.
    return vectorstore.similarity_search(question, k=k)


def format_context(documents):
    blocks = []

    for i, doc in enumerate(documents, start=1):
        metadata = getattr(doc, "metadata", {}) or {}

        blocks.append(
            "\n".join([
                f"[Context chunk {i}]",
                f"Company: {metadata.get('company', 'Unknown')}",
                f"Report year: {metadata.get('year', 'Unknown')}",
                f"Page: {metadata.get('page_number', metadata.get('page', 'Unknown'))}",
                f"File: {_doc_filename(doc)}",
                "",
                (getattr(doc, "page_content", "") or "").strip(),
            ])
        )

    return "\n\n---\n\n".join(blocks)


def get_sources(documents):
    sources = []
    seen = set()

    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        page = metadata.get("page_number", metadata.get("page", "Unknown"))
        filename = _doc_filename(doc) or "Unknown"

        key = (filename, str(page))
        if key in seen:
            continue

        seen.add(key)
        sources.append({
            "company": metadata.get("company", "Unknown"),
            "year": metadata.get("year", "Unknown"),
            "page": page,
            "filename": filename,
        })

    return sources


def _build_comparison_prompt(question, company_docs):
    parts = []

    for company, docs in company_docs.items():
        parts.append(
            f"=== {company} evidence ===\n"
            f"{format_context(docs)}"
        )

    return (
        "\n\n".join(parts)
        + f"\n\nComparison question:\n{question}\n\n"
        "Return a compact comparison. Give one clearly labeled value per company. "
        "If one company's value cannot be supported, say that only for that company."
    )


def ask_question(question, vectorstore, client, messages):
    standalone_question = rewrite_question(
        client,
        question,
        messages,
    )

    companies = detect_companies(standalone_question)

    # Multi-company comparison path
    if len(companies) > 1:
        company_docs = {}
        all_docs = []

        for company in companies:
            docs = retrieve_documents_for_company(
                vectorstore,
                standalone_question,
                company,
                k=5,
            )
            company_docs[company] = docs
            all_docs.extend(docs)

        prompt = _build_comparison_prompt(
            standalone_question,
            company_docs,
        )

        answer = call_llm(
            client,
            [
                {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.1,
        )

        return {
            "answer": answer or FALLBACK_ANSWER,
            "sources": get_sources(all_docs),
            "standalone_question": standalone_question,
            "documents": all_docs,
        }

    # Single-company path
    docs = retrieve_documents(
        vectorstore,
        standalone_question,
        k=6,
    )

    if not docs:
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "standalone_question": standalone_question,
            "documents": [],
        }

    prompt = (
        f"Annual-report context:\n\n{format_context(docs)}\n\n"
        f"Question:\n{standalone_question}\n\n"
        "Answer using only the supplied context."
    )

    answer = call_llm(
        client,
        [
            {"role": "system", "content": FINANCIAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
        temperature=0.1,
    )

    return {
        "answer": answer or FALLBACK_ANSWER,
        "sources": get_sources(docs),
        "standalone_question": standalone_question,
        "documents": docs,
    }
