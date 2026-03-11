from rank_bm25 import BM25Okapi

documents = []
metadatas = []

bm25 = None

def build_index(docs, metas):
    global documents, metadatas, bm25

    documents = [doc.split() for doc in docs]
    metadatas = metas

    bm25 = BM25Okapi(documents)

def keyword_search(query, top_k=5):
    global bm25, documents

    if bm25 is None:
        return [], []
    
    tokenized_query = query.split()

    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        list(enumerate(scores)),
        key=lambda x: x[1],
        reverse=True
    )

    results = ranked[:top_k]

    docs = [" ".join(documents[i]) for i, _ in results]
    metas = [metadatas[i] for i, _ in results]

    return docs, metas