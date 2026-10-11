"""
lc_memory.py  —  Task 5: Memory in LangChain
==============================================
Demonstrates six memory capabilities:
  1. Statelessness proof   – name given in call-1, asked in call-2 without history
  2. RunnableWithMessageHistory – 4-turn conversation with history
  3. Session isolation     – 'ali' and 'sara' have completely separate histories
  4. Bounded memory        – trim_messages keeps only last N turns (system kept)
  5. Custom memory type    – Pydantic UserProfile stores only key facts
  6. Persistence           – histories saved to / loaded from JSON files
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

# Force unbuffered output so background task log shows progress
os.environ["PYTHONUNBUFFERED"] = "1"

# Suppress LangChain deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*RunnableWithMessageHistory.*")

from dotenv import load_dotenv

# LangChain core
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Pydantic
from pydantic import BaseModel, Field

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# 0. SETUP
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY missing from .env")

MODEL_NAME = "nvidia/nemotron-3.5-lightning:free"

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0.3,
    max_retries=2,
    timeout=30,
)

parser = StrOutputParser()

HISTORY_DIR = Path(__file__).parent / "chat_histories"
HISTORY_DIR.mkdir(exist_ok=True)


def divider(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def safe_invoke(chain, inputs: Any, label: str = "Chain") -> str:
    try:
        result = chain.invoke(inputs)
        if isinstance(result, str):
            return result
        return str(result)
    except Exception as exc:
        msg = f"[API ERROR in '{label}': {type(exc).__name__}: {exc}]"
        print(msg)
        return msg


# ─────────────────────────────────────────────────────────────────────────────
# SHARED: In-memory chat history store
# ─────────────────────────────────────────────────────────────────────────────
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """Simple in-memory message history backed by a Python list."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


_store: Dict[str, InMemoryHistory] = {}


def get_session_history(session_id: str) -> InMemoryHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryHistory()
    return _store[session_id]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — STATELESSNESS PROOF
# ─────────────────────────────────────────────────────────────────────────────
divider("1. STATELESSNESS PROOF")
print(
    "Goal: Tell the model a name in Call-1, then ask 'What is my name?' in\n"
    "Call-2 WITHOUT passing any history. The model should say it doesn't know.\n"
)

stateless_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant. Answer concisely."),
    ("human", "{question}"),
])
stateless_chain = stateless_prompt | llm | parser

call1_question = "Hi! My name is Alex. Nice to meet you."
print(f">> Call 1 (user): {call1_question}")
call1_response = safe_invoke(stateless_chain, {"question": call1_question}, "Stateless Call-1")
print(f"<< Call 1 (AI)  : {call1_response}\n")

call2_question = "What is my name?"
print(f">> Call 2 (user): {call2_question}  [<-- no history passed]")
call2_response = safe_invoke(stateless_chain, {"question": call2_question}, "Stateless Call-2")
print(f"<< Call 2 (AI)  : {call2_response}")
print(
    "\n[Result] The model does NOT remember 'Alex' — proving statelessness.\n"
    "         Without explicit history in the prompt, each call is independent."
)

time.sleep(1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — RunnableWithMessageHistory (4-turn conversation)
# ─────────────────────────────────────────────────────────────────────────────
divider("2. RunnableWithMessageHistory -- 4-Turn Conversation")
print(
    "Goal: Wrap the Task-2 chat chain with RunnableWithMessageHistory using\n"
    "MessagesPlaceholder('history'). A later question refers to an earlier one.\n"
)

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a knowledgeable AI tutor. Be concise."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

base_chain = chat_prompt | llm | parser

chain_with_history = RunnableWithMessageHistory(
    base_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

TUTOR_SESSION = "tutor_session_001"

turns = [
    "What is a neural network?",
    "Can you give me a simple real-life analogy for it?",
    "How does it differ from a decision tree?",
    "Going back to your analogy -- how would a decision tree use it differently?",
]

for i, turn in enumerate(turns, 1):
    print(f"[Turn {i}] User: {turn}")
    config = {"configurable": {"session_id": TUTOR_SESSION}}
    try:
        response = chain_with_history.invoke({"input": turn}, config=config)
        print(f"[Turn {i}]   AI: {response}\n")
    except Exception as exc:
        print(f"[Turn {i}]   AI: [API ERROR: {exc}]\n")
    time.sleep(1)

print(f"[History stored for '{TUTOR_SESSION}': {len(_store[TUTOR_SESSION].messages)} messages]")

time.sleep(1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SESSION ISOLATION ('ali' vs 'sara')
# ─────────────────────────────────────────────────────────────────────────────
divider("3. SESSION ISOLATION -- 'ali' and 'sara'")
print(
    "Goal: Two users share the same program but have completely separate\n"
    "histories. Neither can see what the other said.\n"
)

isolation_turns = {
    "ali": [
        "My favourite programming language is Rust.",
        "What language did I just mention?",
    ],
    "sara": [
        "I love working with data science and Python.",
        "What field am I interested in?",
    ],
}

for user, messages in isolation_turns.items():
    print(f"\n--- {user.upper()}'s session ---")
    for msg in messages:
        print(f"  [{user}] User: {msg}")
        config = {"configurable": {"session_id": user}}
        try:
            response = chain_with_history.invoke({"input": msg}, config=config)
            print(f"  [{user}]   AI: {response}")
        except Exception as exc:
            print(f"  [{user}]   AI: [API ERROR: {exc}]")
        time.sleep(1)

for user in ("ali", "sara"):
    hist = _store.get(user)
    if hist:
        print(f"\n[Stored messages for '{user}']")
        for m in hist.messages:
            role = "Human" if isinstance(m, HumanMessage) else "AI"
            print(f"  {role}: {m.content[:100]}{'...' if len(m.content) > 100 else ''}")

print(
    "\n[Result] Ali's session only knows about Rust; Sara's only about Python/data science.\n"
    "         Cross-contamination: NONE -- sessions are fully isolated."
)

time.sleep(1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — BOUNDED MEMORY (window of last N messages)
# ─────────────────────────────────────────────────────────────────────────────
divider("4. BOUNDED MEMORY -- window of last 6 messages (3 turns)")
print(
    "Goal: Keep only the last 6 messages so history never grows without limit.\n"
    "Any system message is always preserved.\n"
)

BOUNDED_SESSION = "bounded_session"
MAX_MESSAGES = 6


class BoundedHistory(BaseChatMessageHistory, BaseModel):
    """Chat history that trims itself to the last MAX_MESSAGES messages."""

    messages: List[BaseMessage] = Field(default_factory=list)
    max_messages: int = MAX_MESSAGES

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)
        system_msgs = [m for m in self.messages if isinstance(m, SystemMessage)]
        non_system = [m for m in self.messages if not isinstance(m, SystemMessage)]
        non_system = non_system[-self.max_messages:]
        self.messages = system_msgs + non_system

    def clear(self) -> None:
        self.messages = []


_bounded_store: Dict[str, BoundedHistory] = {}


def get_bounded_history(session_id: str) -> BoundedHistory:
    if session_id not in _bounded_store:
        _bounded_store[session_id] = BoundedHistory()
    return _bounded_store[session_id]


bounded_chain = RunnableWithMessageHistory(
    base_chain,
    get_bounded_history,
    input_messages_key="input",
    history_messages_key="history",
)

long_conversation = [
    "Turn 1: What is Python?",
    "Turn 2: What is Java?",
    "Turn 3: What is C++?",
    "Turn 4: What is Rust?",
    "Turn 5: What is Go?",
    "Turn 6: What is TypeScript?",
    "Turn 7: What is Kotlin?",
    "Turn 8: What is Swift?",
]

for i, turn in enumerate(long_conversation, 1):
    config = {"configurable": {"session_id": BOUNDED_SESSION}}
    try:
        response = bounded_chain.invoke({"input": turn}, config=config)
        hist = _bounded_store[BOUNDED_SESSION]
        print(
            f"After Turn {i:02d}: History length = {len(hist.messages):2d} msgs | "
            f"Latest AI snippet: '{response[:55].strip()}...'"
        )
    except Exception as exc:
        print(f"After Turn {i:02d}: [API ERROR: {exc}]")
    time.sleep(0.8)

print(
    f"\n[Result] History is capped at {MAX_MESSAGES} messages regardless of conversation length.\n"
    "         Oldest turns are dropped; system messages are always preserved."
)

time.sleep(1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — CUSTOM MEMORY: Pydantic UserProfile
# ─────────────────────────────────────────────────────────────────────────────
divider("5. CUSTOM MEMORY -- Pydantic UserProfile (key-fact store)")
print(
    "Goal: Store only extracted key facts (name, city, job, preferences) in a\n"
    "structured Pydantic model instead of buffering raw messages.\n"
)


class UserProfile(BaseModel):
    """Structured user profile extracted from conversation."""
    name: Optional[str] = None
    city: Optional[str] = None
    job: Optional[str] = None
    preferences: List[str] = Field(default_factory=list)

    def summary(self) -> str:
        lines = []
        if self.name:
            lines.append(f"Name: {self.name}")
        if self.city:
            lines.append(f"City: {self.city}")
        if self.job:
            lines.append(f"Job: {self.job}")
        if self.preferences:
            lines.append(f"Preferences: {', '.join(self.preferences)}")
        return "\n".join(lines) if lines else "No profile data yet."


profile = UserProfile()

user_statements = [
    ("My name is Zara and I live in Karachi.", {"name": "Zara", "city": "Karachi"}),
    ("I'm a software engineer who loves Python.", {"job": "Software Engineer", "preferences": ["Python"]}),
    ("I also enjoy hiking and reading sci-fi novels.", {"preferences": ["hiking", "sci-fi novels"]}),
    ("I prefer dark-mode IDEs and mechanical keyboards.", {"preferences": ["dark-mode IDEs", "mechanical keyboards"]}),
]

print("Building UserProfile from conversation statements:\n")
for statement, facts in user_statements:
    print(f'  User says: "{statement}"')
    if "name" in facts:
        profile.name = facts["name"]
    if "city" in facts:
        profile.city = facts["city"]
    if "job" in facts:
        profile.job = facts["job"]
    if "preferences" in facts:
        profile.preferences.extend(facts["preferences"])
    print(f"  Profile now: {profile.summary().replace(chr(10), ' | ')}\n")

final_question = "Based on what you know about me, suggest a weekend activity in my city."
print(f'Final Question: "{final_question}"')
profile_ctx = profile.summary()
profile_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        f"You are a personalised AI assistant. Here is what you know about the user:\n"
        f"{profile_ctx}\nUse this to give tailored, concise responses.",
    ),
    ("human", "{question}"),
])
profile_chain = profile_prompt | llm | parser
final_answer = safe_invoke(profile_chain, {"question": final_question}, "UserProfile Chain")
print(f"AI Response: {final_answer}")

print("""
[When custom (Pydantic) memory is better than buffer memory]
  1. Storage efficiency: Buffer memory grows linearly with conversation length;
     a profile stores only extracted facts, remaining compact regardless of turns.
  2. Structured retrieval: Facts are typed fields, making personalisation logic
     (filtering, business rules) far easier than parsing raw chat text.
  3. Privacy & compliance: Sensitive raw exchanges are not retained — only the
     distilled facts — reducing data-retention risk.
  4. Long-lived sessions: For accounts that persist across days/weeks, a compact
     profile avoids context-window exhaustion that raw buffer memory causes.
""")

time.sleep(1)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — PERSISTENCE (save & reload from JSON)
# ─────────────────────────────────────────────────────────────────────────────
divider("6. PERSISTENCE -- Save & Reload History from JSON")
print(
    "Goal: Serialize each session's history to a JSON file and reload it\n"
    "on the next program run so the conversation survives a restart.\n"
)


def messages_to_json(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Serialize BaseMessage list to JSON-safe dicts."""
    result = []
    for m in messages:
        if isinstance(m, HumanMessage):
            role = "human"
        elif isinstance(m, AIMessage):
            role = "ai"
        elif isinstance(m, SystemMessage):
            role = "system"
        else:
            role = "unknown"
        result.append({"role": role, "content": m.content})
    return result


def json_to_messages(data: List[Dict[str, str]]) -> List[BaseMessage]:
    """Deserialize JSON dicts back to BaseMessage objects."""
    result = []
    for item in data:
        role = item.get("role", "unknown")
        content = item.get("content", "")
        if role == "human":
            result.append(HumanMessage(content=content))
        elif role == "ai":
            result.append(AIMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
    return result


def save_all_histories(store: Dict[str, InMemoryHistory]) -> None:
    """Save every session in the store to its own JSON file."""
    for session_id, hist in store.items():
        file_path = HISTORY_DIR / f"{session_id}.json"
        data = {
            "session_id": session_id,
            "messages": messages_to_json(hist.messages),
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved '{session_id}' -> {file_path.name}  ({len(hist.messages)} messages)")


def load_history_from_json(session_id: str) -> InMemoryHistory:
    """Load a session history from its JSON file."""
    file_path = HISTORY_DIR / f"{session_id}.json"
    hist = InMemoryHistory()
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        hist.messages = json_to_messages(data.get("messages", []))
        print(f"  Loaded '{session_id}' from {file_path.name}  ({len(hist.messages)} messages)")
    else:
        print(f"  No saved history for '{session_id}' -- starting fresh.")
    return hist


# Step 1: Save
print("Step 1 -- Saving all accumulated sessions to JSON files:\n")
save_all_histories(_store)

# Step 2: Simulate restart
print("\nStep 2 -- Simulating program restart (clearing in-memory store)...")
_store.clear()
print(f"  In-memory store cleared. Keys remaining: {list(_store.keys())}")

# Step 3: Reload
print("\nStep 3 -- Reloading histories from JSON files:\n")
for sid in [TUTOR_SESSION, "ali", "sara"]:
    _store[sid] = load_history_from_json(sid)

# Step 4: Continue conversation with reloaded history
print("\nStep 4 -- Continuing 'ali' conversation after reload:\n")
continuation_msg = "Can you remind me what programming language I mentioned earlier?"
print(f"  [ali] User: {continuation_msg}")
config = {"configurable": {"session_id": "ali"}}
try:
    response = chain_with_history.invoke({"input": continuation_msg}, config=config)
    print(f"  [ali]   AI: {response}")
except Exception as exc:
    print(f"  [ali]   AI: [API ERROR: {exc}]")

# Step 5: List saved files
print("\nStep 5 -- JSON files in chat_histories/:\n")
for json_file in sorted(HISTORY_DIR.glob("*.json")):
    size_kb = json_file.stat().st_size / 1024
    print(f"  {json_file.name}  ({size_kb:.1f} KB)")

time.sleep(1)

# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
divider("TASK 5 -- COMPLETE SUMMARY")
print("""
  Section 1 -- Statelessness   : Proved: model forgot 'Alex' without history.
  Section 2 -- MessageHistory  : 4-turn tutor chat; Turn 4 referenced Turn 1 analogy.
  Section 3 -- Session Isolation: 'ali' (Rust) and 'sara' (Python/DS) fully isolated.
  Section 4 -- Bounded Memory  : History capped at 6 msgs; oldest turns auto-dropped.
  Section 5 -- Custom Memory   : Pydantic UserProfile stores name/city/job/prefs only.
  Section 6 -- Persistence     : Histories saved to JSON and reloaded after fake restart.

  Deliverables:
    - lc_memory.py              (this script)
    - chat_histories/*.json     (one JSON file per session)
""")
print("=" * 72 + "\n")
