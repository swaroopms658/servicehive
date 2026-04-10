# AutoStream Conversational Agent

This project implements the ServiceHive assignment for a social-to-lead conversational AI workflow. The agent answers AutoStream pricing and policy questions from a local knowledge base, detects high-intent leads, collects lead details across multiple turns, and triggers a guarded mock lead capture tool only when all required fields are present.

## How to Run

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_key_here
```

Optional:

```bash
OPENAI_MODEL=gpt-4o-mini
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

The app uses OpenAI as primary and falls back to Groq automatically if OpenAI fails and a `GROQ_API_KEY` is present.

4. Start the CLI:

```bash
python main.py
```



https://github.com/user-attachments/assets/8eadec81-819a-4f8c-9447-1cb63b7050f9



## Architecture

I used LangGraph because this assignment is a stateful workflow rather than a single prompt-response loop. The agent needs to detect intent, retrieve local knowledge, track lead fields across turns, and enforce a strict tool-calling condition before lead capture. LangGraph fits that shape well because each step can be represented as a node with explicit transitions and shared typed state.

State is stored in a LangGraph `MemorySaver` checkpoint and includes conversation messages, current intent, retrieved context, collected lead fields, and whether lead capture has already happened. This allows the agent to retain memory across 5 to 6 turns while keeping business logic deterministic. The RAG layer uses a local Markdown knowledge base with BM25 retrieval, which keeps the setup simple and fully local. Intent classification and slot extraction use structured LLM outputs, while the actual tool execution is guarded in Python code so it never runs before `name`, `email`, and `platform` are all collected.

## WhatsApp Webhook Integration

To deploy this on WhatsApp, I would place the LangGraph app behind a small webhook service using FastAPI . WhatsApp messages would arrive through the Meta WhatsApp Business API webhook. The webhook handler would map each incoming phone number to a stable conversation `thread_id`, pass the user message into the LangGraph workflow, and send the generated reply back through the WhatsApp send-message endpoint. Persistent storage could replace in-memory checkpoints so conversations survive restarts. I would also add webhook signature validation, retry-safe message handling, delivery logging, and background processing for outbound API calls to make the integration production-safe.
