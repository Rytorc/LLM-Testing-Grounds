from app.core.ollama_client import generate
from app.tools.document_tools import list_documents, read_document, search_sources
from .tool_formatter import (
    format_document_list,
    format_source_matches,
    format_document_content,
)

def execute_tool_action(parsed_action):
    tool_name = parsed_action.get("tool")
    args = parsed_action.get("args", {})

    if tool_name == "list_documents":
        documents = list_documents()
        return format_document_list(documents)
    
    if tool_name == "search_sources":
        query = args.get("query", "")
        matches = search_sources(query)
        return format_source_matches(matches, query)
    
    if tool_name == "read_document":
        source = args.get("source", "")
        content = read_document(source)
        return format_document_content(source , content)
    
    return None

def decide_action(user_input, documents, model):
    documents_text = "\n".join(f"- {doc}" for doc in documents)

    prompt = f"""
    You are a routing assistant for a chatbot.

    Available actions:
    1. tool:list_documents
        - Use when the user wants to see all indexed documents.

    2.tool:search_sources:<query>
        - Use when the user wants to find documents by topic, keyword, filename, or subject.

    3. tool:read_document:<source>
        - Use when the user wants to open, show, read, or inspect one specific document.
        - Only use an exact source from the indexed documents list below.

    4. rag
        - Use when the user is asking a normal question that should be answered with retrieval and reasoning.

    Indexed documents:
    {documents_text}

    Rules:
    - Return only one line.
    - Do not explain your answer.
    - If the user is asking to list all docs, return exactly: tool:list_documents
    - If the user wants one specific document and there is a clear exact mathc in the indexed documents, return: tool:read_document:<source>
    - If the user is asking to find documents by topic, return tool:search_sources:<query>
    - If unsure, return: rag

    User input:
    {user_input}

    Action:
    """

    result = generate(model, prompt).strip()
    return result

def parse_action(action_text):
    action_text = action_text.strip()

    if action_text == "rag":
        return {"type": "rag"}
    
    if action_text == "tool:list_documents":
        return {"type": "tool", "tool": "list_documents"}
    
    if action_text.startswith("tool:search_sources:"):
        query = action_text[len("tool:search_sources:"):].strip()
        if not query:
            return {"type": "rag"}
        return {
            "type": "tool",
            "tool": "search_sources",
            "args": {"query": query}
        }
    
    if action_text.startswith("tool:read_document:"):
        source = action_text[len("tool:read_document:"):].strip()
        if not source:
            return {"type": "rag"}
        return {
            "type": "tool",
            "tool": "read_document",
            "args": {"source": source}
        }
    
    return {"type": "rag"}