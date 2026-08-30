from __future__ import annotations

import json
import re
from pathlib import Path

from rag_core import (
    ask_question,
    get_vectorstore,
    load_llm,
    retrieve_documents,
)


QUESTIONS_FILE = Path(
    "evaluation_questions.json"
)

RESULTS_FILE = Path(
    "evaluation_results.json"
)

REFUSAL_PATTERNS = [
    "could not determine",
    "cannot determine",
    "unable to determine",
    "not enough information",
    "not provided in the",
    "do not include",
    "does not include",
]


def normalize_text(
    text: str,
) -> str:
    text = str(
        text or ""
    ).lower()

    text = text.replace(
        "\u00a0",
        " ",
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def compact_number_text(
    text: str,
) -> str:
    return (
        normalize_text(text)
        .replace(",", "")
    )


def contains_all_terms(
    text: str,
    terms: list[str],
) -> bool:
    normalized = normalize_text(
        text
    )

    return all(
        normalize_text(term)
        in normalized
        for term in terms
    )


def contains_any_term(
    text: str,
    terms: list[str],
) -> bool:
    if not terms:
        return True

    normalized = (
        compact_number_text(
            text
        )
    )

    return any(
        compact_number_text(term)
        in normalized
        for term in terms
    )


def contains_forbidden_terms(
    text: str,
    terms: list[str],
) -> bool:
    if not terms:
        return False

    normalized = normalize_text(
        text
    )

    return any(
        normalize_text(term)
        in normalized
        for term in terms
    )


def is_grounded_refusal(
    answer: str,
) -> bool:
    normalized = normalize_text(
        answer
    )

    return any(
        pattern in normalized
        for pattern in REFUSAL_PATTERNS
    )


def extract_money_values_million(
    answer: str,
) -> list[dict]:
    pattern = re.compile(
        r"(?P<currency>\$|€|usd|eur)\s*"
        r"(?P<number>"
        r"\d{1,3}(?:,\d{3})*(?:\.\d+)?"
        r"|\d+(?:\.\d+)?"
        r")"
        r"\s*(?P<unit>"
        r"billion|bn|million|mn|m"
        r")",
        re.IGNORECASE,
    )

    values = []

    for match in pattern.finditer(
        str(answer or "")
    ):
        raw_currency = (
            match.group(
                "currency"
            ).lower()
        )

        currency = (
            "USD"
            if raw_currency in {
                "$",
                "usd",
            }
            else "EUR"
        )

        number = float(
            match.group(
                "number"
            ).replace(
                ",",
                "",
            )
        )

        unit = match.group(
            "unit"
        ).lower()

        if unit in {
            "billion",
            "bn",
        }:
            value_million = (
                number * 1000.0
            )
        else:
            value_million = (
                number
            )

        values.append(
            {
                "currency": (
                    currency
                ),
                "value_million": (
                    value_million
                ),
                "raw": (
                    match.group(0)
                ),
            }
        )

    return values


def money_answer_matches(
    case: dict,
    answer: str,
) -> tuple[
    bool,
    list[dict],
]:
    values = (
        extract_money_values_million(
            answer
        )
    )

    expected_currency = (
        case[
            "expected_currency"
        ]
    )

    expected_value = float(
        case[
            "expected_value_million"
        ]
    )

    tolerance = float(
        case.get(
            "tolerance_million",
            0,
        )
    )

    for item in values:
        if (
            item["currency"]
            != expected_currency
        ):
            continue

        if abs(
            item["value_million"]
            - expected_value
        ) <= tolerance:
            return (
                True,
                values,
            )

    return (
        False,
        values,
    )


def document_filename(
    doc,
) -> str:
    metadata = (
        getattr(
            doc,
            "metadata",
            {},
        )
        or {}
    )

    filename = metadata.get(
        "filename"
    )

    if filename:
        return str(
            filename
        )

    source = str(
        metadata.get(
            "source",
            "",
        )
    )

    return (
        Path(source).name
        if source
        else ""
    )


def retrieval_evidence_matches(
    case: dict,
    documents: list,
) -> tuple[
    bool,
    dict,
]:
    if (
        case["type"]
        == "refusal"
    ):
        return (
            True,
            {
                "evidence_found": None,
                "allowed_source_found": None,
            },
        )

    retrieved_text = "\n".join(
        (
            getattr(
                doc,
                "page_content",
                "",
            )
            or ""
        )
        for doc in documents
    )

    evidence_found = (
        contains_any_term(
            retrieved_text,
            case.get(
                "evidence_text_any",
                [],
            ),
        )
    )

    filenames = {
        document_filename(doc)
        for doc in documents
    }

    allowed = set(
        case.get(
            "allowed_source_filenames",
            [],
        )
    )

    source_found = bool(
        filenames.intersection(
            allowed
        )
    )

    return (
        (
            evidence_found
            and source_found
        ),
        {
            "evidence_found": (
                evidence_found
            ),
            "allowed_source_found": (
                source_found
            ),
            "retrieved_filenames": (
                sorted(
                    filenames
                )
            ),
        },
    )


def returned_source_matches(
    case: dict,
    sources: list[dict],
) -> bool:
    if (
        case["type"]
        == "refusal"
    ):
        return True

    allowed = set(
        case.get(
            "allowed_source_filenames",
            [],
        )
    )

    returned = {
        str(
            source.get(
                "filename",
                "",
            )
        )
        for source in (
            sources
            or []
        )
    }

    return bool(
        returned.intersection(
            allowed
        )
    )


def evaluate_generation(
    case: dict,
    answer: str,
) -> dict:
    required_ok = (
        contains_all_terms(
            answer,
            case.get(
                "required_answer_terms",
                [],
            ),
        )
    )

    if (
        case["type"]
        == "refusal"
    ):
        refusal_ok = (
            is_grounded_refusal(
                answer
            )
        )

        return {
            "generation_pass": (
                refusal_ok
                and required_ok
            ),
            "refusal_ok": (
                refusal_ok
            ),
            "required_terms_ok": (
                required_ok
            ),
            "forbidden_terms_present": False,
            "money_ok": None,
            "money_values_seen": [],
        }

    forbidden = (
        contains_forbidden_terms(
            answer,
            case.get(
                "forbidden_answer_terms",
                [],
            ),
        )
    )

    money_ok = None
    money_values = []

    if (
        case["type"]
        == "money"
    ):
        (
            money_ok,
            money_values,
        ) = (
            money_answer_matches(
                case,
                answer,
            )
        )

        type_ok = money_ok
    else:
        type_ok = True

    generation_pass = (
        required_ok
        and type_ok
        and not forbidden
    )

    return {
        "generation_pass": (
            generation_pass
        ),
        "refusal_ok": None,
        "required_terms_ok": (
            required_ok
        ),
        "forbidden_terms_present": (
            forbidden
        ),
        "money_ok": (
            money_ok
        ),
        "money_values_seen": (
            money_values
        ),
    }


def evaluate_case(
    case: dict,
    vectorstore,
    client,
) -> dict:
    question = case[
        "question"
    ]

    documents = (
        retrieve_documents(
            vectorstore,
            question,
            k=6,
        )
    )

    (
        retrieval_pass,
        retrieval_details,
    ) = (
        retrieval_evidence_matches(
            case,
            documents,
        )
    )

    response = (
        ask_question(
            question,
            vectorstore,
            client,
            [
                {
                    "role": "user",
                    "content": (
                        question
                    ),
                }
            ],
        )
    )

    answer = response.get(
        "answer",
        "",
    )

    sources = response.get(
        "sources",
        [],
    )

    generation = (
        evaluate_generation(
            case,
            answer,
        )
    )

    source_pass = (
        returned_source_matches(
            case,
            sources,
        )
    )

    overall = (
        retrieval_pass
        and generation[
            "generation_pass"
        ]
        and source_pass
    )

    return {
        "id": case["id"],
        "question": question,
        "overall_pass": overall,
        "retrieval_pass": (
            retrieval_pass
        ),
        "generation_pass": (
            generation[
                "generation_pass"
            ]
        ),
        "source_pass": (
            source_pass
        ),
        "retrieval_details": (
            retrieval_details
        ),
        "generation_details": (
            generation
        ),
        "answer": answer,
        "sources": sources,
    }


def print_result(
    result: dict,
):
    status = (
        "PASS"
        if result[
            "overall_pass"
        ]
        else "FAIL"
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        f"{status}: "
        f"{result['id']}"
    )

    print(
        "=" * 88
    )

    print(
        f"Question:   "
        f"{result['question']}"
    )

    print(
        "Retrieval:  "
        + (
            "PASS"
            if result[
                "retrieval_pass"
            ]
            else "FAIL"
        )
    )

    print(
        "Generation: "
        + (
            "PASS"
            if result[
                "generation_pass"
            ]
            else "FAIL"
        )
    )

    print(
        "Sources:    "
        + (
            "PASS"
            if result[
                "source_pass"
            ]
            else "FAIL"
        )
    )

    print(
        "\nAnswer:"
    )

    print(
        result["answer"]
    )

    print(
        "\nReturned sources:"
    )

    if result["sources"]:
        for source in (
            result["sources"]
        ):
            print(
                "  - "
                f"{source.get('company')} | "
                f"{source.get('year')} | "
                f"page {source.get('page')} | "
                f"{source.get('filename')}"
            )
    else:
        print(
            "  (none)"
        )

    if not result[
        "overall_pass"
    ]:
        print(
            "\nFailure diagnostics:"
        )

        print(
            json.dumps(
                {
                    "retrieval": (
                        result[
                            "retrieval_details"
                        ]
                    ),
                    "generation": (
                        result[
                            "generation_details"
                        ]
                    ),
                },
                indent=2,
                default=str,
            )
        )


def main():
    cases = json.loads(
        QUESTIONS_FILE
        .read_text(
            encoding="utf-8"
        )
    )

    print(
        "Loading FAISS vector store..."
    )

    vectorstore = (
        get_vectorstore()
    )

    print(
        "Loading LLM client..."
    )

    client = load_llm()

    results = []

    for case in cases:
        result = (
            evaluate_case(
                case,
                vectorstore,
                client,
            )
        )

        results.append(
            result
        )

        print_result(
            result
        )

    total = len(
        results
    )

    overall = sum(
        result[
            "overall_pass"
        ]
        for result in results
    )

    retrieval = sum(
        result[
            "retrieval_pass"
        ]
        for result in results
    )

    generation = sum(
        result[
            "generation_pass"
        ]
        for result in results
    )

    sources = sum(
        result[
            "source_pass"
        ]
        for result in results
    )

    print(
        "\n"
        + "=" * 88
    )

    print(
        "FINAL EVALUATION SUMMARY"
    )

    print(
        "=" * 88
    )

    print(
        f"Overall:    "
        f"{overall}/{total}"
    )

    print(
        f"Retrieval:  "
        f"{retrieval}/{total}"
    )

    print(
        f"Generation: "
        f"{generation}/{total}"
    )

    print(
        f"Sources:    "
        f"{sources}/{total}"
    )

    RESULTS_FILE.write_text(
        json.dumps(
            results,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if overall != total:
        failed = [
            result["id"]
            for result in results
            if not result[
                "overall_pass"
            ]
        ]

        print(
            "\nFailed cases:"
        )

        for case_id in failed:
            print(
                f"  - {case_id}"
            )

        raise SystemExit(1)

    print(
        "\nAll verified evaluation cases passed."
    )


if __name__ == "__main__":
    main()
