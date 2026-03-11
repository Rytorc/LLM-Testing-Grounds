from app.core.ollama_client import generate

def rewrite_query(query, model):
    prompt = f"""
    Rewrite the user's question so it is clearer and better for document search.

    Do not answer the question.
    Only rewrite it.

    User question:
    {query}

    Rewritten question
    """

    rewritten = generate(model, prompt)

    return rewritten.strip()

def generate_multi_queries(query, model , n=4):
    prompt = f"""
    Generate {n} differnt search queries for retrieving documents relevant to the user's question.

    Rules:
    - Do not answer the question.
    - Each query should be short and retrieval-friendly
    - Use different wording or angles.
    - Return one query per line only.
    - No numbering.

    User question:
    {query}

    Search queries:
    """

    raw_output = generate(model, prompt)

    queries = []
    for line in raw_output.splitlines():
        cleaned = line.strip().lstrip("-").strip()
        if cleaned:
            queries.append(cleaned)

    #deduplicate while preserving order
    seen = set()
    unique_queries = []
    for q in queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique_queries.append(q)

    return unique_queries[:n]