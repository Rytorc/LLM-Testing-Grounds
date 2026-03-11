import os
from dataclasses import dataclass
from dotenv import load_dotenv

APP_ENV = os.getenv("APP_ENV", "development")

if APP_ENV == "production":
    load_dotenv(".env.production")
else:
    load_dotenv(".env")

@dataclass
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"

    model_name: str = os.getenv("MODEL_NAME", "llama3.1")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

    docs_path: str = os.getenv("DOCS_PATH", "test_documents")
    data_path: str = os.getenv("DATA_PATH", "data")
    history_file: str = os.getenv("HISTORY_FILE", "history.json")
    chroma_path: str = os.getenv("CHORMA_PATH", "data/chroma")

    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    multi_query_count: int = int(os.getenv("MULTI_QUERY_COUNT", "4"))
    vector_search_k: int = int(os.getenv("VECTOR_SEARCH_K", "5"))
    keyword_search_k: int = int(os.getenv("KEYWORD_SEARCH_K", "5"))

    max_messages: int = int(os.getenv("MAX_MESSAGES", "10"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "300"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

settings = Settings()