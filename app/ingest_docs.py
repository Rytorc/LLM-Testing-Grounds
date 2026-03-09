from rag import collection, embedder, chunk_text
from document_loader import load_document
import os

DOCS_PATH = "test_documents/"

for root, dirs, files in os.walk(DOCS_PATH):

    for file in files:

        filepath = os.path.join(root, file)

        if not os.path.isfile(filepath):
            continue

        relative_path = os.path.relpath(filepath, DOCS_PATH)

        print(f"Ingesting: {file}")

        collection.delete(where={"source": file})

        text, ext = load_document(filepath)

        chunks = chunk_text(text)

        print(f"{relative_path}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
        
            if not chunk.strip():
                continue

            embedding = embedder.encode(chunk).tolist()

            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": relative_path,
                    "type": ext,
                    "chunk": i
                }],
                ids=[f"{relative_path}_{i}"]
            )

        #LEGACY: Better to separate to chunks
        #add_document(text)

print("Documents Indexed")