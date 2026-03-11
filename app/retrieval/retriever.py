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

def add_document(text):

    embedding = embedder.encode(text).tolist()
    
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(hash(text))]
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

def search_multi_query(queries, top_k=3, vector_k=5, keyword_k=5):
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
    
    pairs = [[queries[0], doc] for doc in merged_docs]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(merged_docs, merged_metas, scores),
        key=lambda x: x[2],
        reverse=True
    )

    top_docs = [item[0] for item in ranked[:top_k]]
    top_metas = [item[1] for item in ranked[:top_k]]

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