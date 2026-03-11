from app.core.memory import ChatMemory
from app.core.ollama_client import generate_stream
from app.core.response_formatter import format_response_with_sources
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

class ChatBot:

    def __init__(self, model=None):
        self.memory = ChatMemory()
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

    def chat(self, user_input):
        available_documents = list_documents()
        action_text = decide_action(user_input, available_documents, self.model)
        parsed_action = parse_action(action_text)

        if parsed_action["type"] == "tool":
            tool_response = execute_tool_action(parsed_action)

            if tool_response is not None:
                print("Bot: ", end="", flush=True)
                print(tool_response)

                self.memory.add_user(user_input)
                self.memory.add_assistant(tool_response)
                self.memory.maybe_compress(self.model)

                return tool_response

        self.memory.add_user(user_input)
        self.memory.maybe_compress(self.model)

        rewritten_query = rewrite_query(user_input, self.model)
        extra_queries = generate_multi_queries(rewritten_query, self.model, n=4)

        queries = [rewritten_query] + [
            q for q in extra_queries if q.lower() != rewritten_query.lower()
        ]

        docs, metadata = search_multi_query(
            queries, 
            settings.retrieval_top_k, 
            settings.vector_search_k, 
            settings.keyword_search_k
        )

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

        Evidence Summary:
        {compressed_context}

        Conversation:
        {memory_prompt}
        """

        print("Bot: ", end="", flush=True)

        draft_reply = generate_stream(self.model, prompt)

        verification = apply_verification(
            user_input=user_input,
            evidence_summary=compressed_context,
            draft_answer=draft_reply,
            model=self.model,
        )

        verified_reply = verification["final_answer"]

        if verified_reply != draft_reply:
            print("\n\n[Answer revised for accuracy]")
            print(verified_reply)

        final_reply, unique_sources, sources_text = format_response_with_sources(verified_reply, metadata)

        if sources_text:
            print(f"\n{sources_text}")

        self.memory.add_assistant(final_reply)
        self.memory.maybe_compress(self.model)

        return final_reply
    
    def save(self):
        self.memory.save_history()