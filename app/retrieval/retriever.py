import hashlib
import chromadb

from sentence_transformers import SentenceTransformer, CrossEncoder
from .keyword_index import keyword_search

from app.config import settings

client = chromadb.PersistentClient(
    path=settings.chroma_path
)

collection = client.get_or_create_collection("documents")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def add_document(text, metadata=None):

    embedding = embedder.encode(text).tolist()
    
    collection.add(
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata or {}],
        ids=[make_chunk_id(text, metadata)],
    )

def search_single_query(query, vector_k=5, keyword_k=5):

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=vector_k
    )

    vector_docs = results["documents"][0] if results["documents"] else []
    vector_meta = results["metadatas"][0] if results["metadatas"] else []

    # Keyword Search
    keyword_docs, keyword_meta = keyword_search(query, top_k=keyword_k)

    # Combine
    docs = vector_docs + keyword_docs
    metas = vector_meta + keyword_meta

    # Remove Duplicates
    seen = set()
    unique_docs = []
    unique_meta = []

    for doc, meta in zip(docs, metas):
        key = (doc.strip(), meta.get("source"), meta.get("chunk"))
        if key not in seen:
            unique_docs.append(doc)
            unique_meta.append(meta)
            seen.add(key)

    return unique_docs, unique_meta

def search_multi_query(queries, top_k=3, vector_k=5, keyword_k=5, return_scores=False):
    all_docs = []
    all_metas = []

    for query in queries:
        docs, metas = search_single_query(
            query,
            vector_k=vector_k,
            keyword_k=keyword_k
        )

        all_docs.extend(docs)
        all_metas.extend(metas)

    seen = set()
    merged_docs = []
    merged_metas = []

    for doc, meta in zip(all_docs, all_metas):
        key = (doc.strip(), meta.get("source"), meta.get("chunk"))
        if key not in seen:
            merged_docs.append(doc)
            merged_metas.append(meta)
            seen.add(key)

    if not merged_docs:
        return [], []

    scored = []
    for doc, meta in zip(merged_docs, merged_metas):
        pairs = [[query, doc] for query in queries]
        scores = reranker.predict(pairs)
        best_score = max(scores) if len(scores) else 0.0
        scored.append((doc, meta, best_score))

    ranked = sorted(
        scored,
        key=lambda x: x[2],
        reverse=True
    )

    top_docs = [item[0] for item in ranked[:top_k]]
    top_metas = [item[1] for item in ranked[:top_k]]

    if return_scores:
        top_scores = [item[2] for item in ranked[:top_k]]
        return top_docs, top_metas, top_scores

    return top_docs, top_metas

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

def make_chunk_id(text, metadata=None):
    source=""
    chunk=""
    if metadata:
        source = str(metadata.get("source", ""))
        chunk = str(metadata.get("chunk", ""))
    raw = f"{source}::{chunk}::{text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()