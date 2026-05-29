from sentence_transformers import SentenceTransformer
import chromadb
import os

# Load embedding model (downloads once, ~90MB)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to local ChromaDB (creates a folder called 'chroma_db')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs")

def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def ingest_file(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    
    chunks = chunk_text(text)
    embeddings = model.encode(chunks).tolist()
    ids = [f"{os.path.basename(filepath)}_chunk_{i}" for i in range(len(chunks))]
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    print(f"Ingested {len(chunks)} chunks from {filepath}")

# Run it
ingest_file("docs/sample.txt")
print("Done! Vector DB ready.")