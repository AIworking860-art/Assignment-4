# Task 7: Mini-RAG FAQ Assistant (`mini_rag.py`)

Bonus Task 7 implements a standalone **Mini Retrieval-Augmented Generation (RAG)** FAQ system powered by two **NVIDIA Nemotron** models via **OpenRouter**:

- **Chat Generation**: `nvidia/nemotron-3.5-lightning:free`
- **Dense Embeddings**: `nvidia/nemotron-3-embed-1b:free`
- **Base Endpoint**: `https://openrouter.ai/api/v1`

---

## Architecture & Features

1. **Knowledge Base (12 FAQ Documents)**:
   - Covers core application queries (e.g., weather lookup, search syntax, input validation limits, and tool constraints).

2. **Embeddings & Vector Retrieval**:
   - Computes dense vector embeddings using `nvidia/nemotron-3-embed-1b:free` via `OpenAIEmbeddings`.
   - Performs pure-Python vector math using dot products and L2 norms (`cosine_similarity`).
   - Retrieves top-$k$ ($k=3$) candidate documents sorted by relevance score.

3. **Relevance Thresholding & Grounded Generation**:
   - Implements a strict confidence threshold (`MIN_SIMILARITY = 0.30`).
   - If the best similarity score falls below this threshold, the model does not hallucinate and immediately responds:
     > *"I couldn't find that information in the knowledge base."*
   - Context is injected directly into a strictly constrained LCEL prompt chain: `prompt | llm | StrOutputParser()`.

4. **Structured JSONL Event Logging**:
   - Every request is logged to `app_log.json` with:
     - Timestamp (ISO 8601 UTC)
     - Session ID
     - User question & Assistant answer
     - Status flag (`success`, `no_relevant_context`, `invalid_input`, `error`)
     - Top similarity score & retrieved document context
     - Model metadata

5. **Validation & Safe Error Handling**:
   - Rejects empty / whitespace-only inputs without wasting model tokens.
   - Catches API rate limits (`429`), network interruptions, and missing credentials safely without crashing.

---

## Setup & Running

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Create a `.env` file in the `Task 7` directory (or copy from `.env.example`):
```env
OPENROUTER_API_KEY="your_openrouter_api_key_here"
```

### 3. Run Automated Tests
```bash
python mini_rag.py --test
```

### 4. Run Interactive Chat
```bash
python mini_rag.py
```
Inside the interactive session:
- Enter your question directly.
- Type `/test` to run the built-in test suite.
- Type `/exit` or `quit` to exit.

---

## Test Cases & Verification Output

The automated test suite evaluates four key scenarios:

| Test Case | Question | Expected Behavior / Output |
| :--- | :--- | :--- |
| **Relevant FAQ: character limit** | *"What is the maximum number of characters accepted by the word-count tool?"* | Matches FAQ-11 with high similarity ($\ge 0.85$); responds: *"The word-count tool accepts a maximum of 20,000 characters per request."* |
| **Relevant FAQ: city search** | *"How can I check the weather for Lahore?"* | Matches FAQ-02 / FAQ-03; explains entering city name to request weather. |
| **Out-of-scope question** | *"Who won the football World Cup in 1986?"* | Similarity below 0.30; responds: *"I couldn't find that information in the knowledge base."* |
| **Empty input** | `"   "` | Handled immediately; responds: *"Please enter a question before submitting."* |
