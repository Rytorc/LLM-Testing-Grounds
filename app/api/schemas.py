from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    used_tool: bool = False
    tool_name: Optional[str] = None
    verification_status: Optional[str]= None

class HealthResponse(BaseModel):
    status: str

class DocumentsResponse(BaseModel):
    documents: List[str]

class ReadDocumentResponse(BaseModel):
    source: str
    content: str