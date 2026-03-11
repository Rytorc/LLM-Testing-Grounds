from app.retrieval.retriever import collection, embedder, chunk_text
from app.retrieval.semantic_chunker import semantic_chunk
from app.retrieval.keyword_index import build_index

from .document_loader import load_document

from app.config import settings

import os

DOCS_PATH = settings.docs_path
all_docs = []
all_metas = []

for root, dirs, files in os.walk(DOCS_PATH):

    for file in files:

        filepath = os.path.join(root, file)

        if not os.path.isfile(filepath):
            continue

        relative_path = os.path.relpath(filepath, DOCS_PATH)

        print(f"Ingesting: {relative_path}")

        collection.delete(where={"source": relative_path})

        text, ext = load_document(filepath)

        chunks = semantic_chunk(text, ext)

        print(f"{relative_path}: {len(chunks)} semantic chunks")

        for i, chunk in enumerate(chunks):
        
            if not chunk.strip():
                continue

            embedding = embedder.encode(chunk).tolist()

            meta = {
                "source": relative_path,
                "type": ext,
                "chunk": i
            }

            all_docs.append(chunk)
            all_metas.append(meta)

            safe_id = relative_path.replace("/", "_")

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[meta],
                ids=[f"{safe_id}_{i}"]
            )

        #LEGACY: Better to separate to chunks
        #add_document(text)

build_index(all_docs, all_metas)
print(f"BM25 indexed docs: {len(all_docs)}")
print("Documents Indexed")