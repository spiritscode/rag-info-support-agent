# RAG Chatbot — Applied AI Demo

A basic RAG system built on Claude API + ChromaDB.
Demonstrates: embeddings, semantic search, context injection, prompt engineering.

## Architecture
User query → embed query → semantic search (ChromaDB) → inject top-k chunks → Claude API → answer

## Setup
pip install -r requirements.txt
cp .env.example .env        # add your Anthropic API key
python ingest.py            # index your documents
python chat.py              # start chatting

## Design decisions
- Chunk size 300 words with 50-word overlap — balances context and precision
- Same embedding model for ingest and query (all-MiniLM-L6-v2) — vector space must match
- History trimmed to last 5 turns — controls token cost without losing context
- "I don't have that information" fallback — prevents hallucination outside the docs
