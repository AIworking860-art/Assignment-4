# Task 6: LCEL Chains in LangChain (`lc_chains.py`)

This project demonstrates core **LangChain Expression Language (LCEL)** patterns using **OpenRouter** and the NVIDIA Nemotron model (`nvidia/nemotron-3.5-lightning:free`).

## Features Demonstrated

1. **Simple Chain (`invoke`, `stream`, `batch`)**:
   - Constructs a basic LCEL pipeline: `prompt | llm | StrOutputParser()`.
   - Tests synchronous single invocation (`.invoke()`), streaming tokens progressively (`.stream()`), and batch queries (`.batch()`).

2. **Sequential Chain (Comparison: Manual vs LCEL Pipeline)**:
   - **Before**: Two manual steps passing intermediate results.
   - **After**: Single unified LCEL chain connected via `RunnableLambda`:
     ```python
     detail_prompt | llm | parser | RunnableLambda(lambda details: {"details": details}) | summary_prompt | llm | parser
     ```

3. **RunnableLambda + RunnablePassthrough**:
   - Uses `RunnablePassthrough.assign()` to dynamically calculate text metrics (word count, character count) and augment dictionary state while preserving original inputs.

4. **Parallel Chain (`RunnableParallel`)**:
   - Executes three independent branches in parallel from a single input:
     - Short English answer
     - Natural Urdu translation branch
     - Keyword extraction branch

5. **Conditional Routing (`RunnableBranch`)**:
   - Classifies user messages into `TECHNICAL`, `STUDY`, `PROJECT`, or `OTHER`.
   - Uses `RunnableBranch` to dynamically route inputs to specialized prompt handlers.

## Setup & Running

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the `Task 6` directory (copy from `.env.example`):
   ```env
   OPENROUTER_API_KEY="your_openrouter_api_key_here"
   ```

3. **Run the script**:
   ```bash
   python lc_chains.py
   ```
