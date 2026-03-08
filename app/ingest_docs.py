from rag import collection, embedder, chunk_text
import os

DOCS_PATH = "test_documents/"

for file in os.listdir(DOCS_PATH):

    filepath = os.path.join(DOCS_PATH, file)

    with open(os.path.join(DOCS_PATH, file)) as f:
        text = f.read()

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):

        embedding = embedder.encode(chunk).tolist()

        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": file,
                "chunk": i
            }],
            ids=[f"{file}_{i}"]
        )

    #LEGACY: Better to separate to chunks
    #add_document(text)

print("Documents Indexed")