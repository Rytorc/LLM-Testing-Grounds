from app.core.ollama_client import generate

def compress_context(query, docs, metadata, model):
    if not docs:
        return ""
    
    context_blocks = []

    for doc, meta in zip(docs, metadata):
        source = meta.get("source", "unknown")
        chunk = meta.get("chunk", "unknown")

        context_blocks.append(
            f"Source: {source}\nChunk: {chunk}\nContent:\n{doc}"
        )

    joined_context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
    You are preparing evidence for a final question-answering assistant

    Your task:
    - Extract only the information relevant to the user's question.
    - Remove repetition.
    - Keep important facts, comands, values, endpoints, versions, and definitions.
    - Preserve source names when useful.
    - Do not answer the user's question directly.
    - Return a concise evidence summary.

    User question:
    {query}

    Retrieved context:
    {joined_context}

    Compressed evidence summary:
    """

    compressed = generate(model, prompt)
    return compressed.strip()