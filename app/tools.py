import os

DOCS_PATH = "test_documents/"

def list_documents():
    documents = []

    for root, dirs, files in os.walk(DOCS_PATH):
        for file in files:
            filepath = os.path.join(root, file)
            if not os.path.isfile(filepath):
                continue

            relative_path = os.path.relpath(filepath, DOCS_PATH)
            documents.append(relative_path)

    documents.sort()
    return documents

def read_document(source):
    filepath = os.path.join(DOCS_PATH, source)

    if not os.path.isfile(filepath):
        return None
    
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"Failed to read document: {e}"
    
def search_sources(query):
    query_lower = query.lower().strip()
    matches = []

    for source in list_documents():
        if query_lower in source.lower():
            matches.append(source)

    return matches