import os
from difflib import SequenceMatcher

from app.config import settings

DOCS_PATH = settings.docs_path

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

def score_source_match(query, source):
    query_lower = query.lower().strip()
    source_lower = source.lower

    score = 0.0

    if query_lower == source_lower:
        score += 100

    if query_lower in source_lower:
        score += 50

    query_tokens = set(query_lower.replace("/", " ").replace(".", " ").split())
    source_tokens = set(source_lower.replace("/", " ").replace(".", " ").split())

    overlap = len(query_tokens & source_tokens)
    score += overlap * 10

    ratio = SequenceMatcher(None, query_lower, source_lower).ratio()
    score += ratio * 25

    return score

def resolve_document_request(query, min_score=20, max_results=5):
    candidates = []

    for source in list_documents():
        score = score_source_match(query,source)
        if score >= min_score:
            candidates.append((source, score))

    candidates.sort(key=lambda item: item[1], reverse=True)

    return candidates[:max_results]