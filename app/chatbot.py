from memory import ChatMemory
from ollama_client import generate_stream
from rag import search_multi_query
from query_rewriter import rewrite_query, generate_multi_queries
from context_compressor import compress_context

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

        sources = "\n".join(
            sorted([m["source"] for m in metadata])
        ) if metadata else ""

        memory_prompt = self.memory.build_prompt()

        prompt = f"""
        You are a helpful assistant.
        Use the provided context if relevant.
        If the evidence is insufficient, say so clearly

        Evidence Summary:
        {compressed_context}

        Sources:
        {sources}

        Conversation:
        {memory_prompt}
        """

        print("Bot: ", end="", flush=True)
        reply = generate_stream(self.model, prompt)

        self.memory.add_assistant(reply)
        self.memory.maybe_compress(self.model)

        return reply
    
    def save(self):
        self.memory.save_history()