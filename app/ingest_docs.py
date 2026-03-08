from rag import add_document
import os

DOCS_PATH = "test_documents/"

for file in os.listdir(DOCS_PATH):

    with open(os.path.join(DOCS_PATH, file)) as f:
        text = f.read()

    add_document(text)
    print("Add Document")

print("Documents Indexed")