def format_document_list(documents):
    if not documents:
        return "No indexed documents found."
    
    lines = ["Indexed documents:"]
    lines.extend(f"- {doc}" for doc in documents)
    return "\n".join(lines)

def format_source_matches(matches, query):
    if not matches:
        return f'No indexed documents matched "{query}".'
    
    lines = [f'Documents matching "{query}":']
    lines.extend(f"- {match}" for match in matches)
    return "\n".join(lines)

def format_document_content(source, content):
    if content is None:
        return f'Document not found: {source}'
    
    return f"""
    Document: {source}

    {content}
    """