from memory import ChatMemory
from ollama_client import generate
from rag import search
from query_rewriter import rewrite_query

class ChatBot:

    def __init__(self, model="llama3.1"):
        self.memory = ChatMemory()
        self.model = model

    def chat(self, user_input):

        self.memory.add_user(user_input)

        # Compress Memory if needed
        self.memory.maybe_compress(self.model)

        rewritten_query = rewrite_query(user_input, self.model)

        docs, metadata = search(rewritten_query)
        context = "\n".join(docs) if docs else ""
        sources = "\n".join([m["source"] for m in metadata]) if metadata else ""

        memory_prompt = self.memory.build_prompt()

        prompt = f"""
        You are a helpful assistant.
        Use the provided context if relevant.

        Context:
        {context}

        Sources:
        {sources}

        Conversation:
        {memory_prompt}
        """

        reply = generate(self.model, prompt)

        self.memory.add_assistant(reply)

        # Compress again for optimization
        self.memory.maybe_compress(self.model)

        return reply
    
    def save(self):
        self.memory.save_history()