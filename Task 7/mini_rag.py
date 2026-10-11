"""Bonus Task 7: Mini-RAG using two NVIDIA Nemotron models via OpenRouter.

Chat generation: nvidia/nemotron-3.5-lightning:free
Embeddings: nvidia/nemotron-3-embed-1b:free
Both models are accessed through OpenRouter using OPENROUTER_API_KEY.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

CHAT_MODEL = "nvidia/nemotron-3.5-lightning:free"
EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b:free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LOG_FILE = Path("app_log.json")
TOP_K = 3
MIN_SIMILARITY = 0.30

# Small FAQ knowledge base (12 documents). Change these to match your own app if required.
FAQS: list[dict[str, str]] = [
    {"id": "FAQ-01", "question": "What is this weather app?", "answer": "It is a weather information app that lets users ask questions about weather features and supported tools."},
    {"id": "FAQ-02", "question": "How do I check the current weather?", "answer": "Enter a city name in the weather search field and submit it to request current weather information."},
    {"id": "FAQ-03", "question": "Can I search weather by city?", "answer": "Yes. The weather lookup accepts a city name, for example Lahore or Karachi."},
    {"id": "FAQ-04", "question": "What information can a weather result show?", "answer": "A weather result may show temperature, weather conditions, humidity, and wind information when the weather service provides them."},
    {"id": "FAQ-05", "question": "Does the app need an internet connection?", "answer": "Yes. Requests to the online weather service require an internet connection."},
    {"id": "FAQ-06", "question": "What happens if the city is not found?", "answer": "The app should show a clear message that the city could not be found and ask the user to check the spelling."},
    {"id": "FAQ-07", "question": "What should I do if the weather API fails?", "answer": "The app should handle the failure safely, show a friendly error message, and allow the user to try again."},
    {"id": "FAQ-08", "question": "Does the app save my location?", "answer": "This sample FAQ does not define location storage. Do not assume a location is saved unless the app explicitly implements that feature."},
    {"id": "FAQ-09", "question": "Can I ask about a forecast?", "answer": "You can ask about forecasts, but the answer depends on whether the connected weather API and app implement forecast data."},
    {"id": "FAQ-10", "question": "How do I report an incorrect result?", "answer": "Check the city spelling and try again. If the result still appears incorrect, report the issue to the app maintainer with the city and time of the request."},
    {"id": "FAQ-11", "question": "What is the maximum number of characters accepted by the word-count tool?", "answer": "The word-count tool accepts a maximum of 20,000 characters per request."},
    {"id": "FAQ-12", "question": "How should the app handle invalid input?", "answer": "Validate user input before making an API request. For empty or invalid input, show a helpful message instead of crashing."},
]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def log_event(event: dict[str, Any]) -> None:
    """Append one JSON event per line to app_log.json (JSONL format)."""
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"[Warning] Could not write log: {exc}")


def build_app():
    """Create model, embeddings, FAQ vectors, and the RAG chain."""
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    llm = ChatOpenAI(
        model=CHAT_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_retries=2,
        timeout=45,
    )
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        # Avoid tokenizer lookup for this provider-specific embedding model.
        check_embedding_ctx_length=False,
        max_retries=2,
    )

    texts = [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in FAQS]
    vectors = embeddings.embed_documents(texts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a careful FAQ assistant. Answer using ONLY the supplied knowledge-base context.
If the context does not contain enough information to answer the question, reply exactly:
"I couldn't find that information in the knowledge base."
Do not invent facts. Keep answers clear and concise.

Knowledge-base context:
{context}"""),
        ("human", "Question: {question}"),
    ])
    rag_chain = prompt | llm | StrOutputParser()
    return embeddings, vectors, texts, rag_chain


def retrieve(question: str, embeddings: OpenAIEmbeddings, vectors: list[list[float]], texts: list[str], top_k: int = TOP_K):
    query_vector = embeddings.embed_query(question)
    scored = [(cosine_similarity(query_vector, vector), i) for i, vector in enumerate(vectors)]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [(score, texts[i]) for score, i in scored[:top_k]]


def answer_question(question: str, app, session_id: str = "default") -> str:
    embeddings, vectors, texts, rag_chain = app
    clean_question = question.strip()
    started = datetime.now(timezone.utc).isoformat()
    if not clean_question:
        response = "Please enter a question before submitting."
        log_event({"timestamp": started, "session_id": session_id, "question": question, "answer": response, "status": "invalid_input", "model": CHAT_MODEL})
        return response

    try:
        matches = retrieve(clean_question, embeddings, vectors, texts)
        best_score = matches[0][0] if matches else 0.0
        if best_score < MIN_SIMILARITY:
            response = "I couldn't find that information in the knowledge base."
            status = "no_relevant_context"
        else:
            context = "\n\n".join(f"[similarity={score:.3f}]\n{text}" for score, text in matches)
            response = rag_chain.invoke({"question": clean_question, "context": context}).strip()
            status = "success"
        log_event({
            "timestamp": started,
            "session_id": session_id,
            "question": clean_question,
            "answer": response,
            "status": status,
            "model": CHAT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_similarity": round(best_score, 4) if matches else None,
            "retrieved_documents": [text for _, text in matches],
        })
        return response
    except Exception as exc:
        response = f"Sorry, the request failed safely ({type(exc).__name__}). Check your OpenRouter API key, internet connection, and whether both selected models are available through OpenRouter."
        log_event({"timestamp": started, "session_id": session_id, "question": clean_question, "answer": response, "status": "error", "error_type": type(exc).__name__, "model": CHAT_MODEL})
        return response


def run_tests(app) -> None:
    """Run four assignment demonstration cases."""
    tests = [
        ("Relevant FAQ: character limit", "What is the maximum number of characters accepted by the word-count tool?"),
        ("Relevant FAQ: city search", "How can I check the weather for Lahore?"),
        ("Out-of-scope question", "Who won the football World Cup in 1986?"),
        ("Empty input", "   "),
    ]
    print("\n" + "=" * 72)
    print("BONUS TASK 7 — MINI-RAG TESTS")
    print(f"Chat model: {CHAT_MODEL}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("=" * 72)
    for label, question in tests:
        print(f"\n[{label}]\nQuestion: {question!r}")
        print("Answer:", answer_question(question, app, session_id="test-session"))


def main() -> None:
    print("Mini-RAG FAQ Assistant")
    print(f"Chat model: {CHAT_MODEL} (OpenRouter)")
    print(f"Embedding model: {EMBEDDING_MODEL} (OpenRouter)")
    print("Preparing knowledge base; first run may take a little time...")
    try:
        app = build_app()
    except Exception as exc:
        print(f"Startup failed: {exc}")
        print("Setup: install requirements and set OPENROUTER_API_KEY in your .env file. Both models are requested through OpenRouter.")
        sys.exit(1)

    if "--test" in sys.argv:
        run_tests(app)
        return

    print("Type a question. Type /test to run test cases, or /exit to quit.\n")
    session_id = "cli-session"
    while True:
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if question.strip().lower() in {"/exit", "exit", "quit"}:
            print("Goodbye!")
            break
        if question.strip().lower() == "/test":
            run_tests(app)
            continue
        print("Assistant:", answer_question(question, app, session_id=session_id), "\n")


if __name__ == "__main__":
    main()
