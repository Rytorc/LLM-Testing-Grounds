from ollama_client import generate

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