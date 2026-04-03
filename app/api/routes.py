from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    DocumentsResponse,
    ReadDocumentResponse,
)
from app.chatbot import ChatBot
from app.tools.document_tools import list_documents, read_document
from app.core.response_formatter import extract_unique_sources

router = APIRouter()

_sessions = {}

def get_session_bot(session_id: str) -> ChatBot:
    if session_id not in _sessions:
        _sessions[session_id] = ChatBot(
            persist_memory=False
        )
    return _sessions[session_id]

@router.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    bot = get_session_bot(request.session_id)
    result = bot.chat_structured(request.message)
    return result

@router.get("/documents", response_model=DocumentsResponse)
def documents():
    docs = list_documents()
    return {"documents": docs}

@router.get("/read-document", response_model=ReadDocumentResponse)
def read_document_route(source: str):
    content = read_document(source)

    if content is None:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "source": source,
        "content": content,
    }