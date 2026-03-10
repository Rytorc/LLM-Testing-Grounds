def extract_unique_sources(metadata):
    unique_sources = []
    seen = set()

    for meta in metadata:
        source = meta.get("source")
        if source and source not in seen:
            seen.add(source)
            unique_sources.append(source)

    return unique_sources

def build_sources_text(unique_sources):
    if not unique_sources:
        return ""
    
    return "Sources: \n" + "\n".join(f"- {source}" for source in unique_sources)

def format_response_with_sources(answer, metadata):
    unique_sources = extract_unique_sources(metadata)
    sources_text = build_sources_text(unique_sources)

    if not sources_text: 
        return answer, unique_sources, ""
    
    final_response = f"{answer}\n\n{sources_text}"
    return final_response, unique_sources, sources_text