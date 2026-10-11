import sys
import os
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows consoles to support Urdu script and emojis
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda, RunnablePassthrough, RunnableParallel, RunnableBranch
)

load_dotenv()
MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"
API_KEY = os.getenv("OPENROUTER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to .env beside lc_chains.py.")

llm = ChatOpenAI(
    model=MODEL_NAME, api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1", temperature=0, max_retries=2
)
parser = StrOutputParser()

def heading(title):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)

# 1. SIMPLE CHAIN: invoke, stream, batch
simple_prompt = ChatPromptTemplate.from_template(
    "Answer clearly in 4-6 sentences. Question: {question}"
)
simple_chain = simple_prompt | llm | parser

heading("1. SIMPLE CHAIN — prompt | llm | StrOutputParser")
question = "What is artificial intelligence? Give two everyday examples."
print("\n.invoke() INPUT:", question)
print(".invoke() RESULT:")
print(simple_chain.invoke({"question": question}))

print("\n.stream() RESULT (chunks as received):")
for chunk in simple_chain.stream({"question": question}):
    print(chunk, end="", flush=True)
print()

print("\n.batch() RESULTS:")
batch_inputs = [
    {"question": "Explain machine learning in simple words."},
    {"question": "What is a chatbot? Give one example."},
    {"question": "What is prompt engineering?"},
]
batch_results = simple_chain.batch(batch_inputs)
for i, result in enumerate(batch_results, 1):
    print(f"Input {i}: {batch_inputs[i-1]['question']}")
    print(f"Output {i}: {result}\n")

# 2. SEQUENTIAL CHAIN — manual BEFORE and LCEL AFTER
detail_prompt = ChatPromptTemplate.from_template(
    "Explain this topic for a beginner in about 100 words: {topic}"
)
summary_prompt = ChatPromptTemplate.from_template(
    "Summarize the explanation below in exactly 3 concise bullet points:\n{details}"
)
detail_chain = detail_prompt | llm | parser
summary_chain = summary_prompt | llm | parser

heading("2A. SEQUENTIAL — BEFORE (two manual calls)")
try:
    details_before = detail_chain.invoke(
        {"topic": "Why structured output is useful in AI applications"}
    )
    print("Intermediate step 1 — detailed answer:\n", details_before)
    summary_before = summary_chain.invoke({"details": details_before})
    print("\nFinal step 2 — summary:\n", summary_before)
except Exception as exc:
    print("Sequential BEFORE error:", type(exc).__name__, str(exc))

heading("2B. SEQUENTIAL — AFTER (one LCEL pipeline)")
sequential_chain = (
    detail_prompt | llm | parser
    | RunnableLambda(lambda details: {"details": details})
    | summary_prompt | llm | parser
)
try:
    print("LCEL chain: detail_prompt | llm | parser | RunnableLambda | summary_prompt | llm | parser")
    print("Final sequential result:")
    print(sequential_chain.invoke(
        {"topic": "Why structured output is useful in AI applications"}
    ))
except Exception as exc:
    print("Sequential AFTER error:", type(exc).__name__, str(exc))

# 3. RunnableLambda + RunnablePassthrough
def calculate_metrics(question_text):
    cleaned = " ".join(question_text.strip().split())
    return {
        "cleaned_question": cleaned,
        "word_count": len(cleaned.split()),
        "character_count": len(cleaned),
    }

answer_prompt = ChatPromptTemplate.from_template(
    "Answer briefly and clearly: {question}"
)
lambda_chain = (
    RunnablePassthrough.assign(
        metrics=RunnableLambda(lambda data: calculate_metrics(data["question"]))
    )
    | RunnablePassthrough.assign(
        answer=(
            RunnableLambda(lambda data: {"question": data["metrics"]["cleaned_question"]})
            | answer_prompt | llm | parser
        )
    )
)
heading("3. RUNNABLELAMBDA + RUNNABLEPASSTHROUGH")
try:
    result = lambda_chain.invoke(
        {"question": "   Explain   why AI needs structured output. "}
    )
    print("Intermediate/final dictionary:", result)
    print("Original question preserved:", result["question"])
    print("Computed word count:", result["metrics"]["word_count"])
    print("Computed character count:", result["metrics"]["character_count"])
    print("Answer:", result["answer"])
except Exception as exc:
    print("RunnableLambda/Passthrough error:", type(exc).__name__, str(exc))

# 4. PARALLEL — three branches, one input
short_prompt = ChatPromptTemplate.from_template(
    "Answer in 2-3 sentences: {question}"
)
translation_prompt = ChatPromptTemplate.from_template(
    "Translate this answer into natural Urdu. Return only the Urdu translation:\n{answer}"
)
keywords_prompt = ChatPromptTemplate.from_template(
    "Extract exactly five keywords from this text. Return comma-separated keywords only:\n{text}"
)
parallel_chain = RunnableParallel(
    short_answer=short_prompt | llm | parser,
    urdu_translation=(
        short_prompt | llm | parser
        | RunnableLambda(lambda answer: {"answer": answer})
        | translation_prompt | llm | parser
    ),
    keywords=(
        RunnableLambda(lambda data: {"text": data["question"]})
        | keywords_prompt | llm | parser
    ),
)
heading("4. PARALLEL CHAIN — RunnableParallel")
try:
    parallel_result = parallel_chain.invoke({
        "question": "Explain how AI assistants can help students study effectively."
    })
    print("Result dictionary:")
    print(parallel_result)
    for key, value in parallel_result.items():
        print(f"\n{key}:\n{value}")
except Exception as exc:
    print("Parallel chain error:", type(exc).__name__, str(exc))

# 5. CONDITIONAL — classify first, then route using RunnableBranch
classifier_prompt = ChatPromptTemplate.from_template(
    "Classify the message into exactly one category: TECHNICAL, STUDY, PROJECT, OTHER. "
    "Return only the uppercase category. Message: {message}"
)
def extract_category(text: str) -> str:
    cleaned = text.strip().upper()
    for cat in ["TECHNICAL", "STUDY", "PROJECT", "OTHER"]:
        if cat in cleaned:
            return cat
    tokens = cleaned.split()
    return tokens[0].strip(".,:") if tokens else "OTHER"

classifier_chain = classifier_prompt | llm | parser | RunnableLambda(extract_category)
technical_chain = ChatPromptTemplate.from_template(
    "You are a technical support assistant. Give practical debugging steps for: {message}"
) | llm | parser
study_chain = ChatPromptTemplate.from_template(
    "You are a study tutor. Explain simply and suggest how to study this: {message}"
) | llm | parser
project_chain = ChatPromptTemplate.from_template(
    "You are a project planner. Suggest a practical plan and next steps for: {message}"
) | llm | parser
default_chain = ChatPromptTemplate.from_template(
    "You are a helpful general assistant. Respond to: {message}"
) | llm | parser

branch = RunnableBranch(
    (lambda x: x["category"] == "TECHNICAL", technical_chain),
    (lambda x: x["category"] == "STUDY", study_chain),
    (lambda x: x["category"] == "PROJECT", project_chain),
    default_chain,
)
conditional_chain = (
    RunnablePassthrough.assign(
        category=RunnableLambda(lambda data: {"message": data["message"]}) | classifier_chain
    )
    | RunnablePassthrough.assign(response=branch)
)
conditional_inputs = [
    {"message": "My Python program fails with ModuleNotFoundError. How do I fix it?"},
    {"message": "Explain the Turing Test in simple words for my exam."},
    {"message": "Help me plan the first version of my personal study assistant app."},
    {"message": "Tell me an interesting fact about the ocean."},
]
heading("5. CONDITIONAL CHAIN — classify then RunnableBranch")
print("Branches: TECHNICAL, STUDY, PROJECT, plus default OTHER.")
for i, item in enumerate(conditional_inputs, 1):
    print(f"\n--- Example {i} ---")
    try:
        category = classifier_chain.invoke({"message": item["message"]})
        print("Original message:", item["message"])
        print("Intermediate classification:", category)
        routed = conditional_chain.invoke(item)
        print("Selected category:", routed["category"])
        print("Final routed response:\n", routed["response"])
    except Exception as exc:
        print("Conditional chain error:", type(exc).__name__, str(exc))

heading("TASK 6 FINISHED")
print("Take terminal screenshots of each section, including intermediate and final outputs.")
