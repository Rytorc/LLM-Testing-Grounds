import re

def chunk_markdown(text):
    pattern = r"(#+ .*)"

    parts = re.split(pattern, text)

    chunks = []
    current = ""

    for part in parts:
        if part.startswith("#"):
            if current.strip():
                chunks.append(current.strip())

            current = part
        else:
            current += "\n" + part

    if current.strip():
        chunks.append(current.strip())

    return chunks

def chunk_paragraphs(text):
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]

def chunk_code(text):
    pattern = r"(def .*?:|class .*?:)"

    parts = re.split(pattern, text)

    chunks = []
    current = ""

    for part in parts:
        if part.startswith("def ") or part.startswith("class "):
            if current.strip():
                chunks.append(current.strip())

            current = part
        else:
            current += "\n" + part

    if current.strip():
        chunks.append(current.strip())

    return chunks

def chunk_fallback(text, size=400):
    words = text.split()

    chunks = []

    for i in range(0, len(words), size):
        chunk = " ".join(words[i:i+size])
        chunks.append(chunk)

    return chunks

def semantic_chunk(text, filetype):
    if filetype in [".md", ".markdown"]:
        return chunk_markdown(text)
    if filetype in [".txt"]:
        return chunk_paragraphs(text)
    if filetype in [".py", ".js", ".cpp", ".java"]:
        return chunk_code(text)
    
    return chunk_fallback(text)