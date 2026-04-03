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
    
    return f"Document: {source}\n\n{content}"

def format_document_not_found(query):
    return f'No indexed document matched "{query}".'

def format_document_candidates(query, candidates):
    if not candidates:
        return format_document_not_found(query)
    
    lines = [f'I found multiple possible documents for "{query}":']
    for source, score in candidates:
        lines.append(f"- {source}")

    lines.append("\nPlease ask again using one of the exact file names.")
    return "\n".join(lines)