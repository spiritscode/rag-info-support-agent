from sentence_transformers import SentenceTransformer
import chromadb
import anthropic

# Same embedding model — MUST be the same as ingest.py
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the same ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs")

# Claude client
claude = anthropic.Anthropic()  

def retrieve(question, top_k=3):
    """Embed the question and find similar chunks"""
    query_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results['documents'][0]  # list of top chunks

def ask(question):
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question)
    context = "\n\n---\n\n".join(chunks)
    
    # Step 2: Build the prompt
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:"""
    
    # Step 3: Call Claude
    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.content[0].text

# Basic chat loop
print("RAG Chatbot ready! Type 'quit' to exit.\n")
while True:
    question = input("You: ").strip()
    if question.lower() == 'quit':
        break
    answer = ask(question)
    print(f"\nBot: {answer}\n")