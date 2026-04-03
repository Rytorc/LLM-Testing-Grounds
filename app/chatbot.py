from app.core.memory import ChatMemory
from app.core.ollama_client import generate_stream, OllamaClientError
from app.core.response_formatter import build_sources_text, extract_unique_sources
from app.core.answer_verifier import apply_verification

from app.retrieval.retriever import search_multi_query
from app.retrieval.query_rewriter import rewrite_query, generate_multi_queries
from app.retrieval.context_compressor import compress_context

from app.tools.document_tools import list_documents, read_document, search_sources
from app.tools.tool_formatter import (
    format_document_list,
    format_source_matches,
    format_document_content
)
from app.tools.tool_router import decide_action, parse_action, execute_tool_action

from .config import settings

@staticmethod
def build_evidence_blocks(docs, metadata):
    blocks = []

    for doc, meta in zip(docs, metadata):
        source = meta.get("source", "unknown")
        chunk = meta.get("chunk", "unknown")
        blocks.append(
            f"Source: {source}\nChunk: {chunk}\nContent:\n{doc}"
        )

    return "\n\n---\n\n".join(blocks)

class ChatBot:

    def __init__(self, model=None, persist_memory=True, history_file=None):
        self.memory = ChatMemory(
            persist=persist_memory,
            history_file=history_file
        )
        self.model = model or settings.model_name

    def try_handle_tool(self, user_input):
        lowered = user_input.strip().lower()

        if lowered in ["list documents", "show documents", "show indexed documents"]:
            documents = list_documents()
            return format_document_list(documents)
        
        if lowered.startswith("read document "):
            source = user_input[len("read document "):].strip()
            content = read_document(source)
            return format_document_content(source, content)
        
        if lowered.startswith("find document "):
            query = user_input[len("find document "):].strip()
            matches = search_sources(query)
            return format_source_matches(matches, query)
        
        if lowered.startswith("search sources "):
            query = user_input[len("search sources "):].strip()
            matches = search_sources(query)
            return format_source_matches(matches, query)
        
        return None

    def chat_structured(self , user_input: str) -> dict:
        available_documents = list_documents()
        action_text = decide_action(user_input, available_documents, self.model)
        parsed_action = parse_action(action_text)

        if parsed_action["type"] == "tool":
            tool_result = execute_tool_action(parsed_action)

            if tool_result is not None:
                tool_name = parsed_action.get("tool")

                self.memory.add_user(user_input)
                self.memory.add_assistant(tool_result["answer"])
                self.memory.maybe_compress(self.model)

                return {
                    "answer": tool_result["answer"],
                    "sources": tool_result.get("sources", []),
                    "used_tool": tool_result.get("used_tool", True),
                    "tool_name": tool_result.get("tool_name"),
                    "verification_status": None,
                }

        self.memory.add_user(user_input)
        self.memory.maybe_compress(self.model)

        rewritten_query = rewrite_query(user_input, self.model)
        extra_queries = generate_multi_queries(rewritten_query, self.model, n=settings.multi_query_count)

        queries = [rewritten_query] + [
            q for q in extra_queries if q.lower() != rewritten_query.lower()
        ]

        docs, metadata, scores = search_multi_query(
            queries, 
            settings.retrieval_top_k, 
            settings.vector_search_k, 
            settings.keyword_search_k,
            return_scores=True
        )

        if not docs or (scores and max(scores) < 0.25):
            fallback = (
                "I couldn't find enough reliable evidence in the indexed documents to answer that confidently."
            )
            self.memory.add_assistant(fallback)
            return {
                "answer": fallback,
                "sources": [],
                "used_tool": False,
                "tool_name": None,
                "verification_status": "UNSUPPORTED"
            }

        raw_evidence = build_evidence_blocks(docs, metadata)
        compressed_context = compress_context(
            rewritten_query,
            docs,
            metadata,
            self.model
        )

        memory_prompt = self.memory.build_prompt()

        prompt = f"""
        You are a helpful technical assistant.

        Answer the user's question using the provided evidence summary.
        Base your answer on the evidence, but do NOT copy it word-for-word unless necessary.

        Important rules:
        - Explain the answer in your own words.
        - Be clear, structured, and helpful.
        - If the user asks for step-by-step help, provide numbered steps.
        - If the user asks for an explanation, teach the concept instead of only repeating commands.
        - Stay grounded in the evidence.
        - Do NOT include a "Sources" section.
        - Do NOT cite sources in your answer.
        - The system will add sources separately.

        User question:
        {user_input}

        Evidence Summary:
        {compressed_context}

        Conversation:
        {memory_prompt}
        """

        try:
            draft_reply = generate_stream(self.model, prompt)
        except OllamaClientError as e:
            fallback = f"Model error: {e}"
            self.memory.add_assistant(fallback)
            return {
                "answer": fallback,
                "sources": [],
                "used_tool": False,
                "tool_name": None,
                "verification_status": "UNSUPPORTED",
            }

        verification = apply_verification(
            user_input=user_input,
            evidence_text=raw_evidence,
            draft_answer=draft_reply,
            model=self.model,
        )

        verified_reply = verification["final_answer"]
        sources = extract_unique_sources(metadata)

        self.memory.add_assistant(verified_reply)
        self.memory.maybe_compress(self.model)

        return {
            "answer": verified_reply,
            "sources": sources,
            "used_tool": False,
            "tool_name": None,
            "verification_status": verification["verification_status"]
        }


    def chat(self, user_input: str) -> str:
        result = self.chat_structured(user_input)

        answer = result["answer"]
        sources = result["sources"]

        sources_text = build_sources_text(sources)
        if sources_text:
            return f"{answer}\n\n{sources_text}"
        
        return answer
    
    def save(self):
        self.memory.save_history()