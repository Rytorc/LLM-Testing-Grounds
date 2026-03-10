from memory import ChatMemory
from ollama_client import generate_stream
from rag import search_multi_query
from query_rewriter import rewrite_query, generate_multi_queries
from context_compressor import compress_context
from response_formatter import format_response_with_sources

class ChatBot:

    def __init__(self, model="llama3.1"):
        self.memory = ChatMemory()
        self.model = model

    def chat(self, user_input):

        self.memory.add_user(user_input)
        self.memory.maybe_compress(self.model)

        rewritten_query = rewrite_query(user_input, self.model)
        extra_queries = generate_multi_queries(rewritten_query, self.model, n=4)

        queries = [rewritten_query] + [
            q for q in extra_queries if q.lower() != rewritten_query.lower()
        ]

        docs, metadata = search_multi_query(queries, top_k=3)

        compressed_context = compress_context(
            rewritten_query,
            docs,
            metadata,
            self.model
        )

        memory_prompt = self.memory.build_prompt()

        prompt = f"""
        You are a helpful assistant.
        Use the provided context if relevant.
        If the evidence is insufficient, say so clearly.

        Important:
        - Do NOT include a "Sources" section.
        - Do NOT cite sources in your answer.
        - The system will add sources separately.

        Evidence Summary:
        {compressed_context}

        Conversation:
        {memory_prompt}
        """

        print("Bot: ", end="", flush=True)
        reply = generate_stream(self.model, prompt)

        final_reply, unique_sources, sources_text = format_response_with_sources(reply, metadata)

        if sources_text:
            print(f"\n{sources_text}")

        self.memory.add_assistant(final_reply)
        self.memory.maybe_compress(self.model)

        return final_reply
    
    def save(self):
        self.memory.save_history()