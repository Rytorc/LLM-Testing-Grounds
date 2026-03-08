import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_or_create_collection("documents")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def add_document(text):

    embedding = embedder.encode(text).tolist()
    
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(hash(text))]
    )

def search(query, top_k=3):

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Prepare pairs for reranker
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, metadatas, scores),
        key=lambda x: x[2],
        reverse=True
    )

    top_docs = [r[0] for r in ranked[:top_k]]
    top_meta = [r[1] for r in ranked[:top_k]]

    return top_docs, top_meta

def chunk_text(text, chunk_size=300, overlap=50):

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    
    return chunks