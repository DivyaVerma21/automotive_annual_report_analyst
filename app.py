import html

import streamlit as st

from rag_core import (
    HUGGINGFACE_PROVIDER,
    ask_question,
    get_vectorstore,
    load_llm,
)


st.set_page_config(
    page_title="Automotive Annual Report Analyst",
    page_icon="📊",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
        color: #111827;
    }

    html,
    body,
    [class*="css"] {
        color: #111827;
    }

    .main-title {
        text-align: center;
        font-size: 2.45rem;
        font-weight: 750;
        color: #101828;
        margin-bottom: 0.25rem;
    }

    .subtitle {
        text-align: center;
        color: #667085;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
    }

    [data-testid="stSidebar"] {
        background-color: #eef2f7;
    }

    [data-testid="stSidebar"] * {
        color: #111827;
    }

    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] strong {
        color: #111827;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #ffffff;
        color: #111827 !important;
        border: 1px solid #d0d5dd;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #667085 !important;
    }

    .stButton > button {
        background-color: #ffffff;
        color: #111827;
        border: 1px solid #d0d5dd;
        border-radius: 8px;
    }

    .stButton > button:hover {
        background-color: #f2f4f7;
        color: #111827;
        border-color: #98a2b3;
    }

    [data-testid="stExpander"] {
        background-color: #ffffff;
        border: 1px solid #e4e7ec;
        border-radius: 10px;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] * {
        color: #111827 !important;
    }

    .source-card {
        padding: 9px 12px;
        background-color: #ffffff;
        color: #1d2939;
        border-left: 4px solid #667085;
        border-radius: 6px;
        margin-bottom: 6px;
        font-size: 0.9rem;
    }

    .architecture-card {
        padding: 11px 12px;
        background-color: #ffffff;
        border: 1px solid #d0d5dd;
        border-radius: 8px;
        font-size: 0.85rem;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


SAMPLE_QUESTIONS = [
    "What was Tesla's total revenue in 2023?",
    "How much revenue did Tesla generate in 2022?",
    "What was Ford's revenue for the year 2020?",
    "What was BMW's profit before tax in 2021?",
]


def render_sources(sources):
    if not sources:
        st.caption(
            "No source pages returned."
        )
        return

    for source in sources:
        company = html.escape(
            str(
                source.get(
                    "company",
                    "Unknown",
                )
            )
        )

        year = html.escape(
            str(
                source.get(
                    "year",
                    "Unknown",
                )
            )
        )

        page = html.escape(
            str(
                source.get(
                    "page",
                    "Unknown",
                )
            )
        )

        filename = html.escape(
            str(
                source.get(
                    "filename",
                    "Unknown",
                )
            )
        )

        st.markdown(
            f"""
            <div class="source-card">
                <strong>{company}</strong> · report {year} · page {page}<br>
                {filename}
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.markdown(
        '<div class="main-title">'
        'Automotive Annual Report Analyst'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Grounded RAG analysis of BMW, Tesla and Ford annual reports'
        '</div>',
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if (
        "pending_question"
        not in st.session_state
    ):
        st.session_state.pending_question = None

    try:
        vectorstore = get_vectorstore()
        client = load_llm()

    except Exception as exc:
        st.error(
            f"Application setup error: {exc}"
        )
        st.stop()

    with st.sidebar:
        st.header(
            "Annual Report Analyst"
        )

        st.write(
            "Available companies:"
        )

        st.markdown(
            """
            - BMW
            - Tesla
            - Ford
            """
        )

        st.divider()

        st.subheader(
            "Example questions"
        )

        for index, sample in enumerate(
            SAMPLE_QUESTIONS
        ):
            if st.button(
                sample,
                key=f"sample_{index}",
                use_container_width=True,
            ):
                (
                    st.session_state
                    .pending_question
                ) = sample

        st.divider()

        if st.button(
            "Clear conversation",
            use_container_width=True,
        ):
            st.session_state.messages = []
            (
                st.session_state
                .pending_question
            ) = None
            st.rerun()

        st.divider()

        st.markdown(
            f"""
            <div class="architecture-card">
            <strong>RAG architecture</strong><br><br>
            Embeddings: MiniLM-L6-v2<br>
            Retrieval: FAISS + lexical reranking<br>
            Generator: Llama 3.1 8B Instruct<br>
            Provider: {html.escape(HUGGINGFACE_PROVIDER)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    for message in (
        st.session_state.messages
    ):
        role = message["role"]

        with st.chat_message(role):
            st.markdown(
                message["content"]
            )

            if (
                role == "assistant"
                and message.get(
                    "sources"
                )
            ):
                with st.expander(
                    "Sources"
                ):
                    render_sources(
                        message[
                            "sources"
                        ]
                    )

                standalone = (
                    message.get(
                        "standalone_question"
                    )
                )

                if standalone:
                    with st.expander(
                        "Retrieval details"
                    ):
                        st.caption(
                            "Standalone retrieval query"
                        )
                        st.code(
                            standalone,
                            language=None,
                        )

    typed_question = st.chat_input(
        "Ask about BMW, Tesla or Ford annual reports..."
    )

    question = (
        typed_question
        or st.session_state.pending_question
    )

    if question:
        (
            st.session_state
            .pending_question
        ) = None

        (
            st.session_state
            .messages
            .append(
                {
                    "role": "user",
                    "content": question,
                }
            )
        )

        with st.chat_message(
            "user"
        ):
            st.markdown(question)

        with st.chat_message(
            "assistant"
        ):
            with st.spinner(
                "Searching annual reports..."
            ):
                try:
                    result = ask_question(
                        question,
                        vectorstore,
                        client,
                        (
                            st.session_state
                            .messages
                        ),
                    )

                    answer = result[
                        "answer"
                    ]

                    st.markdown(
                        answer
                    )

                    if result[
                        "sources"
                    ]:
                        with st.expander(
                            "Sources"
                        ):
                            render_sources(
                                result[
                                    "sources"
                                ]
                            )

                    with st.expander(
                        "Retrieval details"
                    ):
                        st.caption(
                            "Standalone retrieval query"
                        )
                        st.code(
                            result[
                                "standalone_question"
                            ],
                            language=None,
                        )

                except Exception as exc:
                    answer = (
                        "An error occurred while answering "
                        f"the question: {exc}"
                    )

                    result = {
                        "sources": [],
                        "standalone_question": (
                            question
                        ),
                    }

                    st.error(
                        answer
                    )

        (
            st.session_state
            .messages
            .append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": result.get(
                        "sources",
                        [],
                    ),
                    "standalone_question": (
                        result.get(
                            "standalone_question",
                            question,
                        )
                    ),
                }
            )
        )


if __name__ == "__main__":
    main()
