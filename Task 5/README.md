# Task 5: Memory in LangChain (`lc_memory.py`)

This project implements conversational memory patterns in LangChain using **OpenRouter** and the **NVIDIA Nemotron** model (`nvidia/nemotron-3-ultra-550b-a55b:free`).

## Features Demonstrated

1. **Statelessness Proof**: Proves that without an explicit memory buffer or history placeholder, LLM invocations are independent and cannot remember facts stated in earlier calls.
2. **RunnableWithMessageHistory**: Implements multi-turn conversation using `RunnableWithMessageHistory` and `MessagesPlaceholder("history")` across a 4-turn interactive tutor session.
3. **Session Isolation**: Demonstrates strict session isolation between two users (`ali` and `sara`) using independent `session_id` identifiers with zero cross-contamination.
4. **Bounded Memory**: Implements a sliding window bounded history (`BoundedHistory`) that caps stored messages to the last $N$ turns while always preserving system instructions.
5. **Custom Memory (Pydantic `UserProfile`)**: Extracts and stores typed key facts (name, city, job, preferences) in a structured Pydantic schema rather than storing raw message buffers, maximizing token efficiency and data privacy.
6. **Persistence**: Serializes all session histories to individual JSON files in `chat_histories/` and reloads them after simulated process restart to continue conversation seamlessly.

## Model Configuration

- **Model**: `nvidia/nemotron-3-ultra-550b-a55b:free`
- **Provider**: OpenRouter (`https://openrouter.ai/api/v1`)
- **Resilience**: Configured with LangChain `with_fallbacks()` for resilient execution.

## Directory Structure

```
Task 5/
├── lc_memory.py               # Main script demonstrating all 6 memory sections
├── requirements.txt           # Python dependencies
├── .env.example               # Template environment configuration
├── .gitignore                 # Excludes .env, virtualenvs, and cache files
├── chat_histories/            # Persistent JSON chat histories
│   ├── ali.json               # Ali's session history
│   ├── sara.json              # Sara's session history
│   └── tutor_session_001.json # Multi-turn tutor conversation history
└── README.md                  # Documentation
```

## Setup & Execution

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   Copy `.env.example` to `.env` and insert your OpenRouter API key:
   ```env
   OPENROUTER_API_KEY="sk-or-v1-..."
   MODEL_NAME="nvidia/nemotron-3-ultra-550b-a55b:free"
   ```

3. **Run the script**:
   ```bash
   python -u lc_memory.py
   ```
