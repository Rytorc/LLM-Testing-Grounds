import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from keyword_index import keyword_search

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

    vector_docs = results["documents"][0]
    vector_meta = results["metadatas"][0]

    # Keyword Search
    keyword_docs, keyword_meta = keyword_search(query, top_k=5)

    # Combine
    docs = vector_docs + keyword_docs
    metas = vector_meta + keyword_meta

    # Remove Duplicates
    seen = set()
    unique_docs = []
    unique_meta = []

    for doc, meta in zip(docs, metas):
        if doc not in seen:
            unique_docs.append(doc)
            unique_meta.append(meta)
            seen.add(doc)

    return unique_docs[:top_k], unique_meta[:top_k]

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