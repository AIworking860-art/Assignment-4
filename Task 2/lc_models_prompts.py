
import os
from datetime import date

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

from langchain_core.output_parsers import StrOutputParser


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is missing. "
        "Add it to your .env file."
    )


# ============================================================
# 2. MODEL SETUP
# ============================================================

MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=500,
)

print("=" * 70)
print("TASK 2 - MODELS & PROMPTS")
print("=" * 70)

print("\nModel:")
print(MODEL_NAME)


# ============================================================
# 3. MODEL INVOKE WITH PLAIN STRING
# ============================================================

print("\n" + "=" * 70)
print("3. INVOKE WITH PLAIN STRING")
print("=" * 70)

question = (
    "What information should a weather assistant provide "
    "when a user asks for weather in Lahore?"
)

response = llm.invoke(question)

print("\nReturn type:")
print(type(response))

print("\nResponse content:")
print(response.content)


# ============================================================
# 4. INVOKE WITH SYSTEM + HUMAN MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("4. INVOKE WITH SYSTEMMESSAGE + HUMANMESSAGE")
print("=" * 70)

messages = [
    SystemMessage(
        content=(
            "You are a helpful and concise weather assistant. "
            "Answer clearly and do not invent live weather data."
        )
    ),
    HumanMessage(
        content="What information is normally included in a weather report?"
    ),
]

response = llm.invoke(messages)

print("\nReturn type:")
print(type(response))

print("\nContent:")
print(response.content)


# ============================================================
# 5. MODEL OPTIONS EXPERIMENT
# ============================================================

def run_experiment(model, prompt, label, runs=3):
    """
    Run the same prompt multiple times and print the results.
    """

    print("\n" + "-" * 70)
    print(label)
    print("-" * 70)

    for i in range(1, runs + 1):
        result = model.invoke(prompt)

        print(f"\nRun {i}:")
        print(result.content)


experiment_prompt = """
You are a weather assistant.

Explain why weather forecasts can change during the day.
Give a short answer in approximately 3 sentences.
"""


# ------------------------------------------------------------
# 5.1 temperature = 0
# ------------------------------------------------------------

llm_temp_0 = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=200,
)

run_experiment(
    llm_temp_0,
    experiment_prompt,
    "EXPERIMENT A - temperature=0",
)


# ------------------------------------------------------------
# 5.2 temperature = 0.9
# ------------------------------------------------------------

llm_temp_09 = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.9,
    max_tokens=200,
)

run_experiment(
    llm_temp_09,
    experiment_prompt,
    "EXPERIMENT B - temperature=0.9",
)


# ------------------------------------------------------------
# 5.3 temperature = 0.8 + seed=42
# ------------------------------------------------------------

llm_seed = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.8,
    seed=42,
    max_tokens=200,
)

run_experiment(
    llm_seed,
    experiment_prompt,
    "EXPERIMENT C - temperature=0.8, seed=42",
)


# ------------------------------------------------------------
# 5.4 Small max_tokens / num_predict equivalent
# ------------------------------------------------------------

llm_small_output = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=30,
)

run_experiment(
    llm_small_output,
    experiment_prompt,
    "EXPERIMENT D - small max_tokens",
)


# ============================================================
# 6. OPTIONS OBSERVATION TABLE
# ============================================================

print("\n" + "=" * 70)
print("OPTIONS OBSERVATION TABLE")
print("=" * 70)

print(
    """
Setting                         Observation
----------------------------------------------------------------------
temperature=0                  More deterministic and consistent answers.

temperature=0.9                More variation in wording and responses.

temperature=0.8, seed=42       Seed attempts to make output reproducible,
                                although exact reproducibility can depend
                                on the provider/model.

small max_tokens               Output becomes short and may be cut off.
"""
)


# ============================================================
# 7. PROMPT TEMPLATE VALIDATION - BUG
# ============================================================

print("\n" + "=" * 70)
print("7. PROMPT TEMPLATE VALIDATION")
print("=" * 70)

print("\nBEFORE - Buggy prompt")

bad_prompt = """
You are a weather assistant.

City: city
Question: {question}
"""

print(bad_prompt)

print(
    """
Problem:
The word 'city' is written as normal text instead of being
defined as a template variable.

The prompt does not properly represent city as a variable.
"""
)


# ============================================================
# 8. FIX USING PromptTemplate
# ============================================================

print("\nAFTER - Fixed PromptTemplate")

weather_prompt = PromptTemplate(
    input_variables=["city", "question"],
    template="""
You are a helpful weather assistant.

City: {city}

User Question:
{question}

Answer clearly and do not invent live weather information.
""",
    validate_template=True,
)

formatted_prompt = weather_prompt.invoke(
    {
        "city": "Lahore",
        "question": "What weather information should I check?"
    }
)

print(formatted_prompt)


# ============================================================
# 9. DELIBERATELY OMIT A VARIABLE
# ============================================================

print("\n" + "-" * 70)
print("DELIBERATE MISSING VARIABLE TEST")
print("-" * 70)

try:

    broken_prompt = PromptTemplate(
        input_variables=["city", "question"],
        template="""
        City: {city}
        Question: {question}
        """,
        validate_template=True,
    )

    # Deliberately omit "question"
    broken_prompt.invoke(
        {
            "city": "Lahore"
        }
    )

except Exception as e:

    print("\nExpected error:")
    print(type(e).__name__)
    print(e)


# ============================================================
# 10. ChatPromptTemplate
# ============================================================

print("\n" + "=" * 70)
print("10. CHATPROMPTTEMPLATE")
print("=" * 70)


chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are {app_name}, a professional weather assistant.

Today's date is {today}.

Your job is to help users understand weather-related
questions. Do not invent live weather information.
"""
        ),

        MessagesPlaceholder(
            variable_name="history"
        ),

        (
            "human",
            """
City: {city}

Question:
{question}
"""
        ),
    ]
)


chat_messages = chat_prompt.invoke(
    {
        "app_name": "WeatherMate",
        "today": str(date.today()),
        "history": [
            HumanMessage(
                content="I am planning a trip to Lahore."
            ),
            SystemMessage(
                content="The user is planning a trip."
            ),
        ],
        "city": "Lahore",
        "question": "What weather information should I check?"
    }
)


print("\nGenerated messages:")

for message in chat_messages.messages:

    print("\nMessage type:")
    print(type(message).__name__)

    print("Content:")
    print(message.content)


# ============================================================
# 11. PARTIAL VARIABLES
# ============================================================

print("\n" + "=" * 70)
print("11. PARTIAL VARIABLES")
print("=" * 70)


partial_chat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are {app_name}, a helpful weather assistant.

Today's date is {today}.
"""
        ),
        (
            "human",
            "Tell me what weather information I should check for {city}."
        ),
    ]
).partial(
    app_name="WeatherMate",
    today=str(date.today()),
)


partial_result = partial_chat_prompt.invoke(
    {
        "city": "Multan"
    }
)


print("\nPartial prompt result:")

for message in partial_result.messages:
    print(message.content)


# ============================================================
# 12. FEW-SHOT CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("12. FEW-SHOT CLASSIFIER")
print("=" * 70)


examples = [
    {
        "question": "What is the weather in Lahore?",
        "category": "needs_tool",
    },
    {
        "question": "What is the temperature in Multan?",
        "category": "needs_tool",
    },
    {
        "question": "Hello, how are you?",
        "category": "general_chat",
    },
    {
        "question": "Explain what a weather forecast is.",
        "category": "general_chat",
    },
    {
        "question": "Write a program to hack a bank account.",
        "category": "off_topic",
    },
    {
        "question": "Tell me about something unrelated to weather.",
        "category": "off_topic",
    },
]


example_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "Question: {question}"
        ),
        (
            "ai",
            "{category}"
        ),
    ]
)


few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    input_variables=["question"],
)


classifier_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an intent classifier for a weather assistant.

Classify the user question into exactly ONE category:

needs_tool
general_chat
off_topic

Definitions:

needs_tool:
Questions requiring weather/API information.

general_chat:
Normal questions that can be answered without
calling the weather API.

off_topic:
Questions outside the assistant's intended scope.

Return ONLY the category name.
"""
        ),

        few_shot_prompt,

        (
            "human",
            "Question: {question}"
        ),
    ]
)


classifier_chain = classifier_prompt | llm | StrOutputParser()


# ============================================================
# 13. TEST CLASSIFIER ON AT LEAST 5 QUESTIONS
# ============================================================

test_questions = [
    "What is the weather in Lahore?",
    "What is the temperature in Karachi?",
    "Hello, how are you?",
    "What is a weather forecast?",
    "Tell me how to build a website.",
]


print("\nClassifier tests:")

for question in test_questions:

    result = classifier_chain.invoke(
        {
            "question": question
        }
    )

    print("\nQuestion:")
    print(question)

    print("Category:")
    print(result.strip())


# ============================================================
# 14. STREAMING
# ============================================================

print("\n" + "=" * 70)
print("14. STREAMING WITH .stream()")
print("=" * 70)


stream_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a concise weather assistant."
        ),
        (
            "human",
            "{question}"
        ),
    ]
)


stream_chain = stream_prompt | llm | StrOutputParser()


print("\nStreaming answer:\n")

for chunk in stream_chain.stream(
    {
        "question": (
            "Explain in simple words why weather "
            "forecasts can change."
        )
    }
):

    print(chunk, end="", flush=True)

print("\n")


# ============================================================
# 15. BATCH
# ============================================================

print("\n" + "=" * 70)
print("15. BATCH WITH .batch()")
print("=" * 70)


batch_questions = [
    {
        "question": "What is a weather forecast?"
    },
    {
        "question": "Why does temperature change during the day?"
    },
    {
        "question": "What does humidity mean?"
    },
]


batch_results = stream_chain.batch(
    batch_questions
)


for i, result in enumerate(batch_results, start=1):

    print(f"\nQuestion {i} result:")
    print(result)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TASK 2 COMPLETED")
print("=" * 70)

print(
    """
Completed requirements:

[✓] Chat model setup
[✓] Plain-string invoke()
[✓] SystemMessage + HumanMessage invoke()
[✓] temperature=0 experiment
[✓] temperature=0.9 experiment
[✓] temperature=0.8 + seed=42 experiment
[✓] Small max_tokens experiment
[✓] PromptTemplate validation
[✓] Deliberate missing-variable error
[✓] ChatPromptTemplate
[✓] Two or more prompt variables
[✓] Partial variables
[✓] MessagesPlaceholder
[✓] Few-shot classifier
[✓] Five classifier test questions
[✓] stream()
[✓] batch()
"""
)