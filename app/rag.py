import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(
    path="data/chroma"
)

collection = client.get_or_create_collection("documents")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def add_document(text):

    embedding = embedder.encode(text).tolist()
    
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(hash(text))]
    )

def search(query):

    query_embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    return results["documents"][0]